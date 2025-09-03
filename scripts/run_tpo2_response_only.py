#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ===== 固定环境与GPU =====
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

if torch.cuda.is_available():
    try:
        torch.cuda.set_device(0)
    except Exception:
        pass

# ===== 路径 =====
BASE_MODEL_PATH = "/home/zceexl3/ai_accountant/models/LLM-Research/Meta-Llama-3-8B"
ADAPTER_PATH    = "/home/zceexl3/ai_accountant/models/tpo_2/adapter-final"

# ===== 单一测试 query =====
QUERY = "How long should I keep my Self Assessment records? in Wales?"

# ===== 与你旧代码一致的提示模版 =====
def build_generation_prompt(instr: str, inp: str = "") -> str:
    if inp:
        return f"### Instruction:\n{instr}\n\n### Input:\n{inp}\n\n### Response:\n"
    return f"### Instruction:\n{instr}\n\n### Response:\n"

def postprocess_response(text: str) -> str:
    """
    1) 只取 ### Response: 之后
    2) 去除多余空白
    3) 轻量去重复：按句号/换行切分，去重再拼接
    """
    parts = text.split("### Response:\n")
    resp = parts[-1] if len(parts) > 1 else text
    resp = resp.strip()

    # 将多行列表式复读压平
    resp = re.sub(r"[ \t]+\n", "\n", resp)
    resp = re.sub(r"\n{3,}", "\n\n", resp)

    # 句子级去重（保持顺序）
    # 分割：句号、问号、感叹号、换行
    segs = re.split(r"(?<=[。！？.!?])\s+|\n+", resp)
    seen = set()
    cleaned = []
    for s in segs:
        s = s.strip()
        if not s:
            continue
        key = re.sub(r"\s+", " ", s)
        if key not in seen:
            seen.add(key)
            cleaned.append(s)
    resp = " ".join(cleaned).strip()

    return resp

def main():
    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # --- Base Model ---
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        device_map=None
    )

    # --- LoRA Adapter (TPO2) ---
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model = model.to("cuda:0")
    model.eval()

    # --- 使用训练/评测时的模版构造 prompt ---
    prompt = build_generation_prompt(QUERY)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    ).to("cuda:0")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,                 # 贪婪解码，与你基准一致，避免温度/采样引发复读
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
            repetition_penalty=1.05          # 轻微抑制复读（保守设置）
        )

    # 解码并抽取 Response 段落 + 去重复
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = postprocess_response(full_text)

    print("\n=== Model Response (TPO2) ===\n")
    print(response)

if __name__ == "__main__":
    main()
