#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_tpo_min_safe_single.py
- 单条指令测试：强制输出 <THOUGHT>...</THOUGHT> 与 <R>...</R>
- 载入 BASE + LoRA(SFT) 并 merge 后生成
- few-shot：含关键税务要点（差旅/临时工作地、Rent-a-Room 取舍、董事借款核销）
- 仅解码新增 tokens，避免提示回显污染解析；解析含兜底
"""

import os
import re
import torch
from typing import List, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from peft import PeftModel

# ===== 路径 =====
BASE_MODEL_PATH = "/home/zceexl3/ai_accountant/models/LLM-Research/Meta-Llama-3-8B"
ADAPTER_PATH    = "/home/zceexl3/ai_accountant/models/lora-sft/checkpoint-100"

# ===== 设备与精度 =====
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DTYPE = (
    torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    else (torch.float16 if torch.cuda.is_available() else torch.float32)
)

# ===== 标签（思考 & 最终回答）=====
TH_BEGIN, TH_END = "<THOUGHT>", "</THOUGHT>"
RESP_BEGIN, RESP_END = "<R>", "</R>"

# ===== Llama3 聊天模板（base 的回退）=====
LLAMA3_CHAT_TEMPLATE = """{% for message in messages -%}
{% if loop.first %}<|begin_of_text|>{% endif -%}
<|start_header_id|>{{ message['role'] }}<|end_header_id|>
{{ message['content'] }}<|eot_id|>
{%- endfor -%}
{%- if add_generation_prompt -%}
<|start_header_id|>assistant<|end_header_id|>
{%- endif -%}"""

# ===== 自定义停止条件：碰到停词即停 =====
class StopOnEnd(StoppingCriteria):
    def __init__(self, tok, stop_str: str, device):
        ids = tok(stop_str, add_special_tokens=False, return_tensors="pt").input_ids
        self.stop_ids = ids[0].to(device)
    def __call__(self, input_ids, scores, **kwargs) -> bool:
        L = len(self.stop_ids)
        if input_ids.shape[1] < L:
            return False
        return torch.equal(input_ids[0, -L:], self.stop_ids)

def stop_on_token(tok, token_str: str, device):
    return StopOnEnd(tok, token_str, device)

# ===== few-shot：贴合 AI Accountant 的风格（合规、简洁、含关键要点）=====
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

    # --- Few-shot 1：Travel expenses（修正版）---
    fs1_user = "[FORMAT EXAMPLE] Are reimbursed travel expenses taxable for a UK employee?"
    fs1_assistant = (
        "Here is my thought process:\n"
        f"{TH_BEGIN}\n"
        "- Distinguish commuting (home↔permanent workplace, taxable) vs business travel (e.g., temporary workplace/client site, allowable).\n"
        "- Temporary workplace tests: 24-month rule + substantial attendance (~40%) can make a site permanent → commuting.\n"
        "- Travel between two workplaces is business travel.\n"
        "- Homeworking: home is a workplace only if required as a base; otherwise home↔work is commuting.\n"
        "- Reimburse allowable expenses → non-taxable; otherwise taxable via PAYE/P11D. AMAP: 45p up to 10k miles, then 25p; below-AMAP can claim MAR.\n"
        f"{TH_END}\n"
        "Here is my response:\n"
        f"{RESP_BEGIN}\n"
        "Reimbursements for **business travel** (e.g., to a **temporary workplace** or **between workplaces**) are generally **non-taxable** when they reflect allowable expenses. "
        "By contrast, **ordinary commuting** (home↔**permanent** workplace) is **taxable**; reimbursing these costs usually triggers **PAYE/NIC** or reporting (P11D/payrolling). "
        "Key checks: the **24-month rule** and whether you attend a site to a **substantial extent** (~40%). "
        "For private cars, HMRC **AMAP** rates (45p/mile first 10k; 25p thereafter) are tax/NIC-free; if paid below AMAP, claim **MAR**.\n"
        "Hint: HMRC **EIM** (travel & temporary workplaces), **ITEPA** travel provisions (high-level).\n"
        f"{RESP_END}"
    )

    # --- Few-shot 2：Rent-a-Room（修正版）---
    fs2_user = "[FORMAT EXAMPLE] Is the Rent-a-Room scheme better than claiming itemised expenses?"
    fs2_assistant = (
        "Here is my thought process:\n"
        f"{TH_BEGIN}\n"
        "- Eligibility: furnished accommodation in main home; threshold £7,500 (or £3,750 each if shared).\n"
        "- ≤ threshold: typically exempt unless you opt out; > threshold: either scheme (receipts−£7,500, no expenses) or normal rules (receipts−allowable expenses).\n"
        "- Cannot combine scheme with itemised expenses; pick lower taxable profit; note admin via tax return.\n"
        f"{TH_END}\n"
        "Here is my response:\n"
        f"{RESP_BEGIN}\n"
        "It depends on your numbers. **Rent-a-Room** covers **furnished** lettings in your **main home** with a **£7,500** annual threshold "
        "(**£3,750 each** if the income is shared). If receipts are **≤ £7,500**, they are typically **exempt automatically** unless you opt out. "
        "If receipts are **> £7,500**, choose between the **scheme** (tax = receipts − £7,500; **no expenses**) or **normal property rules** (tax = receipts − **allowable expenses**). "
        "Choose the route with the **lower taxable profit**; you **cannot** use both for the same income.\n"
        "Hint: HMRC **PIM** (Rent-a-Room & property income, high-level).\n"
        f"{RESP_END}"
    )

    # --- Few-shot 3：Director’s loan write-off（修正版）---
    fs3_user = "[FORMAT EXAMPLE] Is a director taxed if the company writes off a loan to them?"
    fs3_assistant = (
        "Here is my thought process:\n"
        f"{TH_BEGIN}\n"
        "- Treatment depends on capacity: employment vs shareholding.\n"
        "- By reason of employment → often **earnings**: PAYE/Class 1 NIC (or benefit rules) may apply.\n"
        "- In shareholder capacity → may be **distribution/dividend** (usually no NIC).\n"
        "- Company side: close company loans to participators can trigger **CTA 2010 s455**; relief on repayment/write-off subject to conditions.\n"
        f"{TH_END}\n"
        "Here is my response:\n"
        f"{RESP_BEGIN}\n"
        "Often **yes**, but it depends on **why** the loan existed. If the write-off is **by reason of employment**, it is commonly treated as **taxable earnings** for the director and may be subject to **PAYE/Class 1 NIC** "
        "(or reportable under the relevant benefit rules). If it arises in the capacity of a **shareholder**, it may be taxed as a **distribution/dividend** instead (generally **no NIC**). "
        "At the company level, close-company loans to participators may create a **s455** charge; relief may be due on repayment/write-off under statutory conditions. "
        "Apply the correct reporting route (**PAYE/P11D or dividend**) based on the facts.\n"
        "Hint: HMRC **EIM** (employment income/benefits), **CTM** (loans to participators) — high-level.\n"
        f"{RESP_END}"
    )

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
    return tok.apply_chat_template(
        build_messages(user_query),
        tokenize=False,
        add_generation_prompt=True,
        chat_template=LLAMA3_CHAT_TEMPLATE,
    )

# ===== 解析输出标签（含兜底）=====
def parse_blocks(text: str) -> Tuple[str, str]:
    text = re.sub(r"\blocalctxuserlocalctx\b", "", text, flags=re.I)
    th_match = re.search(re.escape(TH_BEGIN) + r"\s*(.*?)\s*" + re.escape(TH_END), text, flags=re.S)
    r_match  = re.search(re.escape(RESP_BEGIN) + r"\s*(.*?)\s*" + re.escape(RESP_END), text, flags=re.S)
    thought = th_match.group(1).strip() if th_match else ""
    final   = r_match.group(1).strip() if r_match else ""
    if not final:
        r_open = re.search(re.escape(RESP_BEGIN) + r"\s*(.*)$", text, flags=re.S)
        if r_open:
            candidate = r_open.group(1).strip()
            candidate = re.split(r"<\|eot_id\|>|<\|end_of_text\|>", candidate)[0].strip()
            candidate = re.split(r"<\|start_header_id\|>.*?<\|end_header_id\|>", candidate)[0].strip()
            final = candidate
    return thought, final

def main():
    # 可通过环境变量 QUERY 覆盖测试问题
    instruction = os.environ.get(
        "QUERY",
        "As a UK resident employee, are reimbursed travel expenses taxable?"
    )

    print("Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print("Loading base model...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=DTYPE,
        attn_implementation="eager",  # 环境允许可改 "flash_attention_2"
        device_map=None
    ).to(device).eval()

    print("Attaching LoRA and merging...")
    peft_model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    model = peft_model.merge_and_unload().to(device).eval()

    prompt = render_prompt(tok, instruction)
    inputs = tok(prompt, return_tensors="pt", add_special_tokens=False).to(device)

    # 仅解码新增 tokens，避免 few-shot 回显污染解析
    input_len = inputs["input_ids"].shape[1]

    stoppers = StoppingCriteriaList([
        StopOnEnd(tok, RESP_END, device),
        stop_on_token(tok, "<|eot_id|>", device),
    ])

    # 需要更稳可改：do_sample=False, temperature=0.0
    gen_kwargs = dict(
        max_new_tokens=512,
        temperature=0.2,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.05,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
        use_cache=True,
        stopping_criteria=stoppers,
    )

    with torch.inference_mode():
        outputs = model.generate(**inputs, **gen_kwargs)

    # 只看新增部分
    new_tokens = outputs[0, input_len:]
    raw = tok.decode(new_tokens, skip_special_tokens=False)

    print("\n================ RAW OUTPUT (new tokens only) ================\n")
    print(raw)

    thought, final = parse_blocks(raw)
    print("\n================ PARSED ====================\n")
    print("THOUGHT:\n", thought or "[EMPTY]")
    print("\nFINAL:\n", final or "[EMPTY]")

if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    main()
