#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
tpo_generate_candidates_batched.py  (Batch accelerated, minimal-safe tweaks)
- Keep original logic: 5 samples per instruction; 3-stage retry (main -> fallback -> ultimate)
- Batch inference per (temperature, top_p)
- Retry only failed items within the batch
- Stop condition: no custom stopper; parse & truncate to </R>
- OOM guard: auto halve batch size and retry
- Minimal tweaks: left padding (decoder-only correct), new SDPA context, bigger default batch, pin_memory, stable batch seed
"""

import os
import re
import json
import time
import html
import math
import hashlib
import warnings
import torch
from typing import List, Tuple, Dict
from pathlib import Path
from datetime import datetime
from contextlib import nullcontext

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Silence harmless NetworkX warning
warnings.filterwarnings("ignore", message="networkx backend defined more than once", category=RuntimeWarning)

# ===== Paths =====
BASE_MODEL_PATH = "/home/zceexl3/ai_accountant/models/LLM-Research/Meta-Llama-3-8B"
ADAPTER_PATH    = "/home/zceexl3/ai_accountant/models/dpo_1/checkpoint-500"  # or outputs path if needed
DATA_PATH       = Path("/home/zceexl3/ai_accountant/data/train.jsonl")

OUTPUT_DIR = Path("/home/zceexl3/ai_accountant/scripts/tpo2/output")
timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / f"thought_response_{timestamp}.jsonl"

# ===== Device / Performance =====
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # use physical GPU 1
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
torch.backends.cuda.matmul.allow_tf32 = True
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DTYPE = (torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
         else (torch.float16 if torch.cuda.is_available() else torch.float32))

# ===== Sampling =====
MAX_NEW_TOKENS = 384
MERGE_LORA = False
TRY_FLASH_ATTN2 = True        # auto-detect flash-attn2
USE_TORCH_COMPILE = True      # enable compile; will gracefully fall back if it fails

sampling_params = [
    (0.7, 0.9),
    (0.8, 0.85),
    (0.9, 0.95),
    (1.0, 0.9),
    (1.1, 0.95),
]
MAX_RETRIES_PER_SAMPLE = 2

# Fallback & ultimate retry settings
FALLBACK_TEMP, FALLBACK_TOPP, FALLBACK_MIN_NEW = 0.85, 0.92, 48
ULTIMATE_TEMP, ULTIMATE_TOPP, ULTIMATE_MIN_NEW = 0.95, 0.97, 64

# ===== Batch size =====
BATCH_SIZE = 64           # larger batch (you have ample VRAM); script will auto halve on OOM
MIN_BATCH_SIZE = 1

# ===== Tags / Template =====
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

# -------- Small helpers (minimal changes) --------

def _stable_seed_from_idxs(idxs: List[int], seed_base: int) -> int:
    """Generate a stable, batch-level seed from indices + base seed."""
    s = ",".join(map(str, idxs)).encode()
    h = int(hashlib.md5(s).hexdigest(), 16) % 1_000_000_007
    return (seed_base * 1315423911 + h) % 2_147_483_647

def build_messages(user_query: str) -> List[dict]:
    system_rule = (
        "You are an assistant for UK individual taxpayers and small businesses. "
        f"Put hidden reasoning strictly inside {TH_BEGIN} ... {TH_END} "
        f"and the user-facing answer strictly inside {RESP_BEGIN} ... {RESP_END}. "
        "Keep the final answer concise, actionable, and aligned with HMRC guidance and UK legislation when relevant. "
        "If you are unsure about a legislation section or SI number, do NOT invent it; use high-level HMRC manual families."
    )
    generic_rule = (
        "Follow EXACTLY this output structure:\n"
        "Here is my thought process:\n"
        f"{TH_BEGIN} 2–6 concise bullets. {TH_END}\n"
        "Here is my response:\n"
        f"{RESP_BEGIN} Concise, stepwise if useful; brief high-level source hints optional. {RESP_END}\n"
        "Do not reveal chain-of-thought outside the thought tags."
    )
    # brief few-shot
    fs1_user = "[FORMAT EXAMPLE] Are reimbursed travel expenses taxable for a UK employee?"
    fs1_assistant = ("Here is my thought process:\n"
                     f"{TH_BEGIN}\n"
                     "- Distinguish commuting vs business travel; temporary workplace tests.\n"
                     "- Reimburse allowable business travel → non-taxable; commuting reimbursements usually taxable.\n"
                     "- AMAP 45p first 10k miles, then 25p; MAR if below AMAP.\n"
                     f"{TH_END}\n"
                     "Here is my response:\n"
                     f"{RESP_BEGIN}\n"
                     "Reimbursements for **business travel** are generally **non-taxable**; commuting is taxable. "
                     "AMAP **45p/mile** (first 10k), **25p** thereafter; claim **MAR** if paid below.\n"
                     "Hint: HMRC **EIM** (high-level).\n"
                     f"{RESP_END}")
    fs2_user = "[FORMAT EXAMPLE] Is the Rent-a-Room scheme better than claiming itemised expenses?"
    fs2_assistant = ("Here is my thought process:\n"
                     f"{TH_BEGIN}\n"
                     "- Eligibility; threshold £7,500 (£3,750 each if shared).\n"
                     "- ≤ threshold exempt; > threshold choose scheme vs normal rules.\n"
                     f"{TH_END}\n"
                     "Here is my response:\n"
                     f"{RESP_BEGIN}\n"
                     "It depends on figures; pick the route with lower taxable profit.\n"
                     "Hint: HMRC **PIM** (high-level).\n"
                     f"{RESP_END}")
    fs3_user = "[FORMAT EXAMPLE] Is a director taxed if the company writes off a loan to them?"
    fs3_assistant = ("Here is my thought process:\n"
                     f"{TH_BEGIN}\n"
                     "- Employment capacity vs shareholder; PAYE/NIC vs dividend.\n"
                     "- Company: CTA 2010 s455 context (high-level).\n"
                     f"{TH_END}\n"
                     "Here is my response:\n"
                     f"{RESP_BEGIN}\n"
                     "Often **yes**, depends on reason; employment-related may be earnings (PAYE/Class 1 NIC). …\n"
                     "Hint: HMRC **EIM/CTM** (high-level).\n"
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
    text = html.unescape(text)
    # take the last matched block; then truncate at known special tokens/headers
    def _last_block(open_tag, close_tag):
        m_all = list(re.finditer(open_tag + r"\s*(.*?)\s*" + close_tag, text, flags=re.S))
        if not m_all:
            return ""
        s = m_all[-1].group(1)
        s = re.split(r"<\|eot_id\|>|<\|end_of_text\|>|<\|start_header_id\|>", s)[0]
        return s.strip()
    th = _last_block(re.escape(TH_BEGIN), re.escape(TH_END))
    r  = _last_block(re.escape(RESP_BEGIN), re.escape(RESP_END))
    if not r:
        m = re.search(re.escape(RESP_BEGIN) + r"\s*(.*)$", text, flags=re.S)
        if m:
            r = re.split(r"<\|eot_id\|>|<\|end_of_text\|>|<\|start_header_id\|>", m.group(1))[0].strip()
    return th, r

def batch_tokenize(tok: AutoTokenizer, prompts: List[str], max_new_tokens: int, model_max_len: int):
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
              max_length=max(8, model_max_len - max_new_tokens), add_special_tokens=False)
    # pin to accelerate H2D
    for k in enc:
        enc[k] = enc[k].pin_memory()
    input_ids = enc["input_ids"]
    attn_mask = enc["attention_mask"]
    pad_id = tok.pad_token_id
    input_lens = [(row != pad_id).sum().item() for row in input_ids]
    return input_ids, attn_mask, input_lens

def batched_generate_texts(model, tok, input_ids, attn_mask, input_lens,
                           temperature, top_p, min_new_tokens, seed_hint=None):
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
        repetition_penalty=1.08,   # slightly stronger anti-repetition
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
        use_cache=True,
    )

    amp_ctx = autocast(device_type="cuda", dtype=torch.bfloat16) if torch.cuda.is_available() else nullcontext()

    # SDPA context (new API only; avoid deprecated torch.backends.cuda.sdp_kernel)
    sdp_ctx = nullcontext()
    try:
        from torch.nn.attention import sdpa_kernel as _sdpa_kernel  # torch >= 2.1
        sdp_ctx = _sdpa_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True)
    except Exception:
        sdp_ctx = nullcontext()

    input_ids = input_ids.to(device, non_blocking=True)
    attn_mask = attn_mask.to(device, non_blocking=True)

    with torch.inference_mode(), amp_ctx, sdp_ctx:
        outputs = model.generate(input_ids=input_ids, attention_mask=attn_mask, **gen_kwargs)

    out_texts = []
    for b in range(outputs.size(0)):
        new_ids = outputs[b, input_lens[b]:]
        out_texts.append(tok.decode(new_ids, skip_special_tokens=False))
    return out_texts

def try_generate_stage(model, tok, prompts, temperature, top_p, min_new_tokens,
                       batch_size: int, seed_base: int) -> List[Tuple[str, str]]:
    results: List[Tuple[str, str]] = [("", "")] * len(prompts)
    idxs = list(range(len(prompts)))
    cur_bs = batch_size

    # for truncation
    model_max_len = getattr(model.config, "max_position_embeddings", 8192)

    while idxs:
        chunk = idxs[:cur_bs]
        sub_prompts = [prompts[k] for k in chunk]

        # tokenize with truncation and LEFT padding (decoder-only correct)
        input_ids, attn_mask, input_lens = batch_tokenize(tok, sub_prompts, MAX_NEW_TOKENS, model_max_len)

        try:
            seed_hint = _stable_seed_from_idxs(chunk, seed_base)
            texts = batched_generate_texts(
                model, tok, input_ids, attn_mask, input_lens,
                temperature, top_p, min_new_tokens, seed_hint=seed_hint
            )
            for local_i, t in enumerate(texts):
                th, resp = parse_blocks(t)
                results[chunk[local_i]] = (th, resp)
            idxs = idxs[cur_bs:]
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if cur_bs == MIN_BATCH_SIZE:
                print("[WARN] OOM at MIN_BATCH_SIZE; attempting per-item retry.")
                for k in chunk:
                    try:
                        ids1, m1, len1 = batch_tokenize(tok, [prompts[k]], MAX_NEW_TOKENS, model_max_len)
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
            print(f"[WARN] Batch generation failed: {e}")
            for k in chunk:
                results[k] = ("", "")
            idxs = idxs[cur_bs:]

    return results

def main():
    print(f"[INFO] Output file: {OUTPUT_PATH}")

    # Load data
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    dataset = []
    for ex in records:
        ins = (ex.get("instruction") or ex.get("question") or ex.get("input") or "").strip()
        if ins:
            dataset.append(ins)
    N = len(dataset)
    print(f"[INFO] Loaded {N} instructions.")

    # Tokenizer
    print("Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"   # IMPORTANT: decoder-only models expect left padding

    # Attention implementation (auto-detect flash-attn2 if available)
    attn_impl = "sdpa"
    if TRY_FLASH_ATTN2:
        try:
            import flash_attn  # type: ignore  # noqa: F401
            attn_impl = "flash_attention_2"
            print("[INFO] flash-attn2 detected; using flash_attention_2.")
        except Exception:
            attn_impl = "sdpa"
            print("[INFO] flash-attn2 not available; using sdpa.")

    # Model
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

    # Main loop
    with open(OUTPUT_PATH, "a", encoding="utf-8") as out_f:
        total_items = 0

        for j, (temperature, top_p) in enumerate(sampling_params):
            print(f"[INFO] Sampling grid {j+1}/{len(sampling_params)}: T={temperature}, top_p={top_p}")

            for start in tqdm(range(0, N, BATCH_SIZE), desc=f"T{temperature}_p{top_p}"):
                end = min(start + BATCH_SIZE, N)
                batch_instructions = dataset[start:end]
                batch_size_now = len(batch_instructions)

                prompts_main = [render_prompt(tok, ins) for ins in batch_instructions]
                prompts_fb   = [render_minimal_prompt(tok, ins) for ins in batch_instructions]

                seed_base = 10_000_000 * (start + 1) + j * 1000

                # Stage 1: main prompt
                stage1 = try_generate_stage(
                    model, tok, prompts_main, temperature, top_p, min_new_tokens=24,
                    batch_size=batch_size_now, seed_base=seed_base
                )

                need_fb = [idx for idx, (_th, _r) in enumerate(stage1) if not _r.strip()]

                # Stage 2: fallback (short prompt)
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

                # Stage 3: ultimate (main prompt, hotter & longer)
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

                # Write out
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
                        "seed_hint": seed_base,     # batch-level seed hint
                        "attn_impl": attn_impl,
                    }
                    out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    total_items += 1

        print(f"✅ Finished: {OUTPUT_PATH}")
        print(f"[INFO] Total items written: {total_items}  (expected {N * len(sampling_params)})")


if __name__ == "__main__":
    main()
