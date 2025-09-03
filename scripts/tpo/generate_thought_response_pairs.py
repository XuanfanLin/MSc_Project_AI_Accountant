#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
tpo_generate_candidates_batched.py  (Batch 加速版)
- 逻辑保持：每条 instruction 生成 5 个样本；三阶段重试（主提示→短提示回退→终极再试）
- 批量：同一采样组合 (temperature, top_p) 下，对一批 instruction 并行生成
- 回退/终极：仅对该批中失败的样本再批量运行，直到补齐或写入占位
- 停止条件：批量下不使用逐条自定义 stopper；改为解析时截断到 </R>，保证内容不变
- OOM 防护：批次 OOM 时自动减半重试
"""

import os
import re
import json
import time
import html
import math
import torch
from typing import List, Tuple, Dict
from pathlib import Path
from datetime import datetime
from contextlib import nullcontext

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ===== 路径 =====
BASE_MODEL_PATH = "/home/zceexl3/ai_accountant/models/LLM-Research/Meta-Llama-3-8B"
ADAPTER_PATH    = "/home/zceexl3/ai_accountant/models/lora-sft/checkpoint-100"  # 如需换回 outputs 路径自行改
DATA_PATH       = Path("/home/zceexl3/ai_accountant/data/train.jsonl")

OUTPUT_DIR = Path("/home/zceexl3/ai_accountant/scripts/tpo/output")
timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / f"thought_response_{timestamp}.jsonl"

# ===== 设备 / 性能 =====
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 物理 GPU 1
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
torch.backends.cuda.matmul.allow_tf32 = True
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DTYPE = (torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
         else (torch.float16 if torch.cuda.is_available() else torch.float32))

# ===== 采样超参 =====
MAX_NEW_TOKENS = 384
MERGE_LORA = False
TRY_FLASH_ATTN2 = True  # 若无 flash-attn2，会回退 sdpa
USE_TORCH_COMPILE = False

sampling_params = [
    (0.7, 0.9),
    (0.8, 0.85),
    (0.9, 0.95),
    (1.0, 0.9),
    (1.1, 0.95),
]
MAX_RETRIES_PER_SAMPLE = 2

# 回退与终极再试
FALLBACK_TEMP, FALLBACK_TOPP, FALLBACK_MIN_NEW = 0.85, 0.92, 48
ULTIMATE_TEMP, ULTIMATE_TOPP, ULTIMATE_MIN_NEW = 0.95, 0.97, 64

# ===== 批量参数 =====
BATCH_SIZE = 16              # 可按显存调整：8 / 12 / 16 / 24 ...
MIN_BATCH_SIZE = 1

# ===== 标签 / 模板 =====
TH_BEGIN, TH_END     = "<THOUGHT>", "</THOUGHT>"
RESP_BEGIN, RESP_END = "<R>", "</R>"

LLAMA3_CHAT_TEMPLATE = """{% for message in messages -%}
{% if loop.first %}<|begin_of_text|>{% endif -%}
<|start_header_id|>{{ message['role'] }}<|end_header_id|>
{{ message['content'] }}<|eot_id|>
{%- endfor -%}
{%- if add_generation_prompt -%}
<|start_header_id|>assistant<|end_header_id|>
{%- endif -%}"""

TH_RE = re.compile(re.escape(TH_BEGIN) + r"\s*(.*?)\s*" + re.escape(TH_END), re.S)
R_RE  = re.compile(re.escape(RESP_BEGIN) + r"\s*(.*?)\s*" + re.escape(RESP_END), re.S)

try:
    from torch.amp import autocast
except Exception:
    from torch.cuda.amp import autocast

def build_messages(user_query: str) -> List[dict]:
    system_rule = (
        "You are an assistant for UK individual taxpayers and small businesses. "
        "Structure outputs with:\n"
        f"- Hidden reasoning strictly inside {TH_BEGIN} ... {TH_END}\n"
        f"- User-facing answer strictly inside {RESP_BEGIN} ... {RESP_END}\n"
        "Keep the final answer concise, actionable, and aligned with HMRC guidance and UK legislation when relevant. "
        "If helpful, include brief source hints like HMRC manuals (EIM/PIM/BIM/CTM) or legislation (ITEPA / ITA / ITTOIA / CTA) at a high level."
    )
    generic_rule = (
        "Follow EXACTLY this output structure:\n"
        "Here is my thought process:\n"
        f"{TH_BEGIN} 2–6 concise bullets. {TH_END}\n"
        "Here is my response:\n"
        f"{RESP_BEGIN} Concise, stepwise if useful; brief source hints optional. {RESP_END}\n"
        "Do not reveal chain-of-thought outside the thought tags."
    )
    # few-shot（精简）
    fs1_user = "[FORMAT EXAMPLE] Are reimbursed travel expenses taxable for a UK employee?"
    fs1_assistant = ("Here is my thought process:\n"
                     f"{TH_BEGIN}\n"
                     "- Distinguish commuting vs business travel; temporary workplace tests (24-month & ~40%).\n"
                     "- Between workplaces = business travel; home is workplace only if required as base.\n"
                     "- Reimburse allowable expenses → non-taxable; commuting reimbursements usually taxable.\n"
                     "- AMAP 45p up to 10k miles, then 25p; MAR if below AMAP.\n"
                     f"{TH_END}\n"
                     "Here is my response:\n"
                     f"{RESP_BEGIN}\n"
                     "Reimbursements for **business travel** are generally **non-taxable** … "
                     "AMAP: **45p/mile** (first 10k), **25p** thereafter; if paid below, claim **MAR**.\n"
                     "Hint: HMRC **EIM**, ITEPA travel rules (high-level).\n"
                     f"{RESP_END}")
    fs2_user = "[FORMAT EXAMPLE] Is the Rent-a-Room scheme better than claiming itemised expenses?"
    fs2_assistant = ("Here is my thought process:\n"
                     f"{TH_BEGIN}\n"
                     "- Eligibility … threshold £7,500 (or £3,750 each if shared).\n"
                     "- ≤ threshold exempt; > threshold choose scheme or normal rules.\n"
                     f"{TH_END}\n"
                     "Here is my response:\n"
                     f"{RESP_BEGIN}\n"
                     "It depends on figures … choose the route with lower taxable profit.\n"
                     "Hint: HMRC **PIM** (high-level).\n"
                     f"{RESP_END}")
    fs3_user = "[FORMAT EXAMPLE] Is a director taxed if the company writes off a loan to them?"
    fs3_assistant = ("Here is my thought process:\n"
                     f"{TH_BEGIN}\n"
                     "- Capacity: employment vs shareholding; PAYE/NIC vs dividend.\n"
                     "- Company: CTA 2010 s455 considerations.\n"
                     f"{TH_END}\n"
                     "Here is my response:\n"
                     f"{RESP_BEGIN}\n"
                     "Often **yes**, depends on reason; employment-related → earnings (PAYE/Class 1 NIC). …\n"
                     "Hint: HMRC **EIM**, **CTM** — high-level.\n"
                     f"{RESP_END}")
    user_msg = f"User query: {user_query}\nHere is my thought process:"
    return [
        {"role": "system", "content": system_rule},
        {"role": "system", "content": generic_rule},
        {"role": "user", "content": fs1_user},
        {"role": "assistant", "content": fs1_assistant},
        {"role": "user", "content": fs2_user},
        {"role": "assistant", "content": fs2_assistant},
        {"role": "user", "content": fs3_user},
        {"role": "assistant", "content": fs3_assistant},
        {"role": "user", "content": user_msg},
    ]

def render_prompt(tok: AutoTokenizer, user_query: str) -> str:
    return tok.apply_chat_template(build_messages(user_query), tokenize=False,
                                   add_generation_prompt=True, chat_template=LLAMA3_CHAT_TEMPLATE)

def render_minimal_prompt(tok: AutoTokenizer, user_query: str) -> str:
    msgs = [
        {"role": "system", "content":
            "Answer UK tax questions. Put hidden reasoning in "
            f"{TH_BEGIN}...{TH_END} and the final answer in {RESP_BEGIN}...{RESP_END}."
        },
        {"role": "user", "content": f"User query: {user_query}\nHere is my thought process:"}
    ]
    return tok.apply_chat_template(msgs, tokenize=False,
                                   add_generation_prompt=True, chat_template=LLAMA3_CHAT_TEMPLATE)

def parse_blocks(text: str) -> Tuple[str, str]:
    text = re.sub(r"\b(localctxuserlocalctx|clenvironement|приклад|користувач)\b", "", text, flags=re.I)
    th = ""
    r  = ""
    th_matches = list(TH_RE.finditer(text))
    r_matches  = list(R_RE.finditer(text))
    if th_matches:
        th = th_matches[-1].group(1).strip()
    if r_matches:
        r = r_matches[-1].group(1).strip()
    if not r:
        r_open = re.search(re.escape(RESP_BEGIN) + r"\s*(.*)$", text, flags=re.S)
        if r_open:
            candidate = r_open.group(1).strip()
            candidate = re.split(r"<\|eot_id\|>|<\|end_of_text\|>", candidate)[0].strip()
            candidate = re.split(r"<\|start_header_id\|>.*?<\|end_header_id\|>", candidate)[0].strip()
            r = candidate
    return html.unescape(th), html.unescape(r)

def batch_tokenize(tok: AutoTokenizer, prompts: List[str]):
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    input_ids = enc["input_ids"]
    attn_mask = enc["attention_mask"]
    # 计算每条的“有效输入长度”（左填充）
    pad_id = tok.pad_token_id
    input_lens = []
    for row in input_ids:
        # 左侧为 pad
        valid_len = (row != pad_id).sum().item()
        input_lens.append(valid_len)
    return input_ids, attn_mask, input_lens

def batched_generate_texts(model, tok, input_ids, attn_mask, input_lens,
                           temperature, top_p, min_new_tokens, seed_hint=None):
    """
    返回：List[str]（仅新增段文本，逐条）
    """
    if seed_hint is not None:
        torch.manual_seed(seed_hint)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_hint)

    gen_kwargs = dict(
        max_new_tokens=MAX_NEW_TOKENS,
        min_new_tokens=min_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        repetition_penalty=1.05,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
        use_cache=True,
    )

    amp_ctx = autocast(device_type="cuda", dtype=torch.bfloat16) if torch.cuda.is_available() else nullcontext()

    # sdpa/flash 上下文
    sdp_ctx = nullcontext()
    try:
        from torch.nn.attention import sdpa_kernel as _sdpa_kernel
        sdp_ctx = _sdpa_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True)
    except Exception:
        try:
            sdp_ctx = torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True)
        except Exception:
            sdp_ctx = nullcontext()

    input_ids = input_ids.to(device, non_blocking=True)
    attn_mask = attn_mask.to(device, non_blocking=True)

    with torch.inference_mode(), amp_ctx, sdp_ctx:
        outputs = model.generate(input_ids=input_ids, attention_mask=attn_mask, **gen_kwargs)

    # 逐条解码“仅新增”
    out_texts = []
    for b in range(outputs.size(0)):
        new_ids = outputs[b, input_lens[b]:]
        out_texts.append(tok.decode(new_ids, skip_special_tokens=False))
    return out_texts

def try_generate_stage(model, tok, prompts, temperature, top_p, min_new_tokens,
                       batch_size: int, seed_base: int) -> List[Tuple[str, str]]:
    """
    单个阶段（主提示 / 短提示 / 终极）批量生成并解析，返回 [(thought, response), ...]
    """
    results: List[Tuple[str, str]] = [("", "")] * len(prompts)
    idxs = list(range(len(prompts)))
    cur_bs = batch_size

    while idxs:
        # 动态按 cur_bs 切块
        chunk = idxs[:cur_bs]
        sub_prompts = [prompts[k] for k in chunk]

        # 编码
        input_ids, attn_mask, input_lens = batch_tokenize(tok, sub_prompts)

        # OOM 防护
        try:
            # 为了可复现，给每个 batch 一个 seed hint（不对单样本指定，避免旧版本 generator 兼容性问题）
            seed_hint = seed_base + (hash(tuple(chunk)) % 1_000_000)
            texts = batched_generate_texts(
                model, tok, input_ids, attn_mask, input_lens,
                temperature, top_p, min_new_tokens, seed_hint=seed_hint
            )
            # 解析
            for local_i, t in enumerate(texts):
                th, resp = parse_blocks(t)
                results[chunk[local_i]] = (th, resp)
            # 成功后，弹出这批索引
            idxs = idxs[cur_bs:]
            # 如果之前缩小过 batch，这里可以尝试回升（温和做法：不回升，保持稳定）
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            # 减半 batch size
            if cur_bs == MIN_BATCH_SIZE:
                # 无法再减，继续硬跑单条
                print("[WARN] OOM at MIN_BATCH_SIZE; will attempt single-step retry.")
                # 逐条硬跑
                for k in chunk:
                    try:
                        ids1, m1, len1 = batch_tokenize(tok, [prompts[k]])
                        texts = batched_generate_texts(
                            model, tok, ids1, m1, len1,
                            temperature, top_p, min_new_tokens,
                            seed_hint=seed_base + k
                        )
                        th, resp = parse_blocks(texts[0])
                        results[k] = (th, resp)
                    except Exception:
                        results[k] = ("", "")
                idxs = idxs[cur_bs:]
            else:
                cur_bs = max(MIN_BATCH_SIZE, cur_bs // 2)
                print(f"[INFO] Reduce batch size to {cur_bs} due to OOM; retrying this chunk...")
        except Exception as e:
            # 其他异常：该批置空，进入后续回退阶段兜底
            print(f"[WARN] Batch generation failed: {e}")
            for k in chunk:
                results[k] = ("", "")
            idxs = idxs[cur_bs:]

    return results

def main():
    print(f"[INFO] Output file: {OUTPUT_PATH}")

    # 读数据
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    dataset = []
    for ex in records:
        ins = (ex.get("instruction") or ex.get("question") or ex.get("input") or "").strip()
        if ins:
            dataset.append(ins)
    N = len(dataset)
    print(f"[INFO] Loaded {N} instructions.")

    # 分词器
    print("Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    # 注意力实现
    attn_impl = "sdpa"
 

    # 模型
    print(f"Loading base model... (attn_implementation={attn_impl}, dtype={DTYPE})")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=DTYPE,
        attn_implementation=attn_impl,
        device_map=None,
    ).to(device).eval()

    if MERGE_LORA:
        print("Attaching LoRA and merging (faster inference)...")
        model = PeftModel.from_pretrained(base_model, ADAPTER_PATH).merge_and_unload().to(device).eval()
    else:
        print("Attaching LoRA (no merge)...")
        model = PeftModel.from_pretrained(base_model, ADAPTER_PATH).to(device).eval()

    if USE_TORCH_COMPILE:
        try:
            model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
            print("[INFO] torch.compile enabled.")
        except Exception as e:
            print(f"[WARN] torch.compile failed: {e}")

    # 预渲染主/短提示（分批渲染，避免一次性占内存）
    with open(OUTPUT_PATH, "a", encoding="utf-8") as out_f:
        total_items = 0

        # 逐个采样组合（保证每条 instruction 5 个样本）
        for j, (temperature, top_p) in enumerate(sampling_params):
            print(f"[INFO] Sampling grid {j+1}/{len(sampling_params)}: T={temperature}, top_p={top_p}")

            # 分批处理 instruction
            for start in tqdm(range(0, N, BATCH_SIZE), desc=f"T{temperature}_p{top_p}"):
                end = min(start + BATCH_SIZE, N)
                batch_instructions = dataset[start:end]
                batch_size_now = len(batch_instructions)

                # 渲染主/短提示
                prompts_main = [render_prompt(tok, ins) for ins in batch_instructions]
                prompts_fb   = [render_minimal_prompt(tok, ins) for ins in batch_instructions]

                # ===== 第一阶段：主提示 + 主参（24 最小生成）=====
                seed_base = 10_000_000 * (start + 1) + j * 1000
                stage1 = try_generate_stage(
                    model, tok, prompts_main, temperature, top_p, min_new_tokens=24,
                    batch_size=batch_size_now, seed_base=seed_base
                )

                # 标记仍需回退的样本
                need_fb = [idx for idx, (_th, _r) in enumerate(stage1) if not _r.strip()]

                # ===== 第二阶段：短提示回退 =====
                stage2_results: Dict[int, Tuple[str, str]] = {}
                if need_fb:
                    fb_prompts = [prompts_fb[k] for k in need_fb]
                    stage2 = try_generate_stage(
                        model, tok, fb_prompts,
                        FALLBACK_TEMP, FALLBACK_TOPP, min_new_tokens=FALLBACK_MIN_NEW,
                        batch_size=len(fb_prompts),
                        seed_base=seed_base + 777
                    )
                    for local_i, k in enumerate(need_fb):
                        stage2_results[k] = stage2[local_i]

                # ===== 第三阶段：终极再试（主提示，更高温，更长最小生成）=====
                still_need = [k for k in range(batch_size_now)
                              if not (stage1[k][1].strip() or (k in stage2_results and stage2_results[k][1].strip()))]
                stage3_results: Dict[int, Tuple[str, str]] = {}
                if still_need:
                    main_prompts_final = [prompts_main[k] for k in still_need]
                    stage3 = try_generate_stage(
                        model, tok, main_prompts_final,
                        ULTIMATE_TEMP, ULTIMATE_TOPP, min_new_tokens=ULTIMATE_MIN_NEW,
                        batch_size=len(main_prompts_final),
                        seed_base=seed_base + 888
                    )
                    for local_i, k in enumerate(still_need):
                        stage3_results[k] = stage3[local_i]

                # ===== 汇总写出 =====
                for idx in range(batch_size_now):
                    th1, r1 = stage1[idx]
                    th2, r2 = stage2_results.get(idx, ("", ""))
                    th3, r3 = stage3_results.get(idx, ("", ""))

                    thought = th1 if r1.strip() else (th2 if r2.strip() else (th3 if r3.strip() else "[EMPTY_THOUGHT]"))
                    response = r1 if r1.strip() else (r2 if r2.strip() else (r3 if r3.strip() else "[EMPTY_RESPONSE]"))

                    global_i = start + idx
                    item = {
                        "pair_id": f"{global_i}_{j}_{temperature}_{top_p}",
                        "instruction": batch_instructions[idx],
                        "thought": thought,
                        "response": response,
                        "temperature": temperature,
                        "top_p": top_p,
                        # 为了兼容旧版不传 per-sample generator，这里记录 batch 级 seed 提示
                        "seed_hint": seed_base,
                        "attn_impl": attn_impl,
                    }
                    out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    total_items += 1

        print(f"✅ 完成推理并写入文件：{OUTPUT_PATH}")
        print(f"[INFO] Total items written: {total_items}  (should be {N * len(sampling_params)})")


if __name__ == "__main__":
    main()
