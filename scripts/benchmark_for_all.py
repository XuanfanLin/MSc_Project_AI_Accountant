#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import json
import time
import torch
import random
import requests
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any, List, Optional
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ================= Env & CUDA =================
# 避免显存碎片
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# 如需强制可见设备（可选）： os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# 固定使用 GPU0
if torch.cuda.is_available():
    try:
        torch.cuda.set_device(0)
    except Exception:
        pass

# ================= Paths & Config =================
BASE_MODEL_PATH  = "/home/zceexl3/ai_accountant/models/LLM-Research/Meta-Llama-3-8B"
SFT_ADAPTER_PATH = "/home/zceexl3/ai_accountant/models/lora-sft/checkpoint-100"
TPO_ADAPTER_PATH = "/home/zceexl3/ai_accountant/models/dpo_1/checkpoint-500"       # 第一次 TPO
TPO2_ADAPTER_PATH= "/home/zceexl3/ai_accountant/models/dpo_2/adapter-final"        # 第二次 TPO

DATA_PATH = Path("/home/zceexl3/ai_accountant/data/test.jsonl")

OUT_DIR = Path("/home/zceexl3/ai_accountant/scripts/overall_benchmark")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PREDICT_OUTPUTS = {
    "base": OUT_DIR / "base_predictions.jsonl",
    "sft":  OUT_DIR / "sft_predictions.jsonl",
    "tpo":  OUT_DIR / "tpo_predictions.jsonl",
    "tpo2": OUT_DIR / "tpo2_predictions.jsonl",
}
SCORE_OUTPUTS = {
    "base": OUT_DIR / "base_scores.jsonl",
    "sft":  OUT_DIR / "sft_scores.jsonl",
    "tpo":  OUT_DIR / "tpo_scores.jsonl",
    "tpo2": OUT_DIR / "tpo2_scores.jsonl",
}
SUMMARY_PATH = OUT_DIR / "summary_scores.json"

# ================= Gemini Config =================
# 已内嵌 API Key；若设置了环境变量 GEMINI_API_KEY，则优先生效
API_KEY = os.environ.get("GEMINI_API_KEY") or "AIzaSyBEie1_V2cXwBrOUe07C3R-gnTPWDhI6nc"
MODEL_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"
HEADERS = {"Content-Type": "application/json"}
SESSION = requests.Session()

# ================= Reproducibility & CUDA opts =================
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ================= Tokenizer (load once) =================
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_PATH, trust_remote_code=True, local_files_only=True
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# ================= Utilities =================
def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def write_jsonl(path: Path, rows: List[Dict[str, Any]], append: bool = False):
    mode = "a" if append else "w"
    with open(path, mode, encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ================= Prompt Builders =================
def build_generation_prompt(instr: str, inp: Optional[str] = "") -> str:
    if inp:
        return f"### Instruction:\n{instr}\n\n### Input:\n{inp}\n\n### Response:\n"
    return f"### Instruction:\n{instr}\n\n### Response:\n"

def build_judge_prompt_no_ref(example: Dict[str, Any]) -> str:
    instruction = example["instruction"].strip()
    pred = example.get("model_output", "").strip()
    return f"""
You are a UK tax expert and impartial judge. Evaluate ONLY the assistant's answer against the question below.
Do NOT assume a ground-truth answer is provided. Judge on:
1) factual accuracy under UK tax rules (flag any likely inaccuracies or hallucinations),
2) legal reasoning and coherence,
3) completeness and actionable guidance (disclaimers OK),
4) clarity and structure.

Return ONLY a JSON object (no code fences) with:
- "score": integer 1–100 (1=very poor, 100=excellent),
- "justification": 1–3 sentences, no newlines.

[Question]
{instruction}

[Assistant Answer]
{pred}
""".strip()

# ================= Model Inference =================
def load_model_with_optional_adapter(base_model_path: str, adapter_path: Optional[str] = None):
    """
    固定在 GPU0：禁用 device_map='auto'，加载到 CPU 后再 .to('cuda:0')
    避免被自动分片到已占满的 GPU1/GPU2。
    """
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        device_map=None  # 关键：不要用 'auto'
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model = model.to("cuda:0")  # 关键：固定到 GPU0
    model.eval()
    return model

@torch.no_grad()
def generate_predictions(label: str, model, records: List[Dict[str, Any]], output_path: Path, batch_size: int = 8):
    print(f"\n🚀 Generating predictions for: {label.upper()}")
    rows = []
    for i in tqdm(range(0, len(records), batch_size)):
        batch = records[i:i+batch_size]
        prompts = [build_generation_prompt(x["instruction"], x.get("input", "")) for x in batch]
        # 关键：张量固定到 GPU0
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        ).to("cuda:0")

        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,  # 基准用贪婪，便于可比
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True
        )
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for j, ex in enumerate(batch):
            text = decoded[j]
            parts = text.split("### Response:\n")
            resp = parts[-1].strip() if len(parts) > 1 else text.strip()
            rows.append({
                "instruction": ex["instruction"],
                "input": ex.get("input", ""),
                "model_output": resp,
            })
    write_jsonl(output_path, rows)
    print(f"✅ Saved predictions: {output_path}")

# ================= Judge Call =================
def call_gemini(prompt: str, max_retries: int = 4, backoff: float = 1.5) -> str:
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.0, "topP": 1.0, "topK": 1, "maxOutputTokens": 256},
    }
    for attempt in range(max_retries):
        try:
            r = SESSION.post(MODEL_URL, headers=HEADERS, data=json.dumps(payload), timeout=60)
            if r.status_code == 200:
                return r.json()["candidates"][0]["content"]["parts"][0]["text"]
            if r.status_code in (429, 500, 503):
                time.sleep(backoff ** attempt)
                continue
            return r.text
        except Exception:
            time.sleep(backoff ** attempt)
    return ""

def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()

def parse_gemini_json(text: str) -> Dict[str, Any]:
    if not text:
        return {"score": None, "justification": ""}
    s = _strip_code_fences(text)
    # 1) 直接 JSON
    try:
        obj = json.loads(s)
        score = obj.get("score")
        if isinstance(score, int) and 1 <= score <= 100:
            just = str(obj.get("justification", "")).replace("\n", " ")
            return {"score": score, "justification": just[:400]}
    except Exception:
        pass
    # 2) 抽取第一个 {} 再解析
    mjson = re.search(r"(\{.*?\})", s, flags=re.S)
    if mjson:
        try:
            obj = json.loads(mjson.group(1))
            score = obj.get("score")
            if isinstance(score, int) and 1 <= score <= 100:
                just = str(obj.get("justification", "")).replace("\n", " ")
                return {"score": score, "justification": just[:400]}
        except Exception:
            pass
    # 3) 兜底：提取一个 1-100 的整数
    m = re.search(r"\b(100|[1-9]?\d)\b", s)
    score = int(m.group(1)) if m else None
    if score is not None and not (1 <= score <= 100):
        score = None
    return {"score": score, "justification": s[:400].replace("\n", " ")}

def evaluate_with_gemini(pred_path: Path, out_path: Path, label: str):
    print(f"\n📐 Evaluating {label.upper()} with Gemini (1–100, no reference)")
    records = read_jsonl(pred_path)
    with open(out_path, "w", encoding="utf-8") as fout:
        for ex in tqdm(records):
            prompt = build_judge_prompt_no_ref(ex)
            raw = call_gemini(prompt)
            parsed = parse_gemini_json(raw)
            ex["judge_raw"] = raw
            ex["judge_score"] = parsed.get("score")
            ex["judge_justification"] = parsed.get("justification")
            fout.write(json.dumps(ex, ensure_ascii=False) + "\n")
            time.sleep(0.05)
    print(f"✅ Saved scores: {out_path}")

def summarize_scores(path: Path) -> Dict[str, Any]:
    rows = read_jsonl(path)
    scores = [r.get("judge_score") for r in rows if isinstance(r.get("judge_score"), int)]
    return {
        "file": str(path),
        "num_scored": len(scores),
        "avg_score": round(sum(scores) / len(scores), 2) if scores else None,
        "min": min(scores) if scores else None,
        "max": max(scores) if scores else None,
    }

# ================= Main =================
if __name__ == "__main__":
    test_records = read_jsonl(DATA_PATH)

    # 统一增删配置，这里会自动跑/汇总
    configs = [
        ("base", None),
        ("sft",  SFT_ADAPTER_PATH),
        ("tpo",  TPO_ADAPTER_PATH),
        ("tpo2", TPO2_ADAPTER_PATH),
    ]

    # 依次生成预测与评分（固定 GPU0）
    for label, adapter in configs:
        model = load_model_with_optional_adapter(BASE_MODEL_PATH, adapter_path=adapter)
        generate_predictions(label, model, test_records, PREDICT_OUTPUTS[label], batch_size=8)
        del model
        torch.cuda.empty_cache()
        evaluate_with_gemini(PREDICT_OUTPUTS[label], SCORE_OUTPUTS[label], label)

    # 汇总
    summary = {
        "meta": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "dataset": str(DATA_PATH),
            "num_test_items": len(test_records),
            "base_model": BASE_MODEL_PATH,
            "sft_adapter": SFT_ADAPTER_PATH,
            "tpo_adapter": TPO_ADAPTER_PATH,
            "tpo2_adapter": TPO2_ADAPTER_PATH,
            "judge": "gemini-2.0-flash (1–100 rubric, no reference)",
        }
    }

    for label, _ in configs:
        summary[label] = summarize_scores(SCORE_OUTPUTS[label])

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n📈 Benchmark summary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nAll artifacts saved under: {OUT_DIR}")
