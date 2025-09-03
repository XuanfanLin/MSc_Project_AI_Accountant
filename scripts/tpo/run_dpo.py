#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, re, json, random, inspect
from pathlib import Path
from typing import List, Dict

# ===== 环境 =====
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")                 # 按需改卡
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from peft import PeftModel
from trl import DPOConfig, DPOTrainer

# ===== 路径 =====
BASE_MODEL_PATH = "/home/zceexl3/ai_accountant/models/LLM-Research/Meta-Llama-3-8B"
ADAPTER_PATH    = "/home/zceexl3/ai_accountant/models/lora-sft/checkpoint-100"
DPO_DATA_PATH   = "/home/zceexl3/ai_accountant/scripts/tpo/output/preference_pairs_gemini.jsonl"
DPO_OUTPUT_DIR  = "/home/zceexl3/ai_accountant/models/tpo_1"

# 参考模型可选（rDPO 省显存）
USE_REFERENCE_FREE = False

GLOBAL_SEED = 42
torch.set_float32_matmul_precision("high")


# === 与 SFT 完全一致的 prompt 形态 ===
def build_plain_prompt_from_instruction(instr: str) -> str:
    instr = (instr or "").strip()
    return f"### Instruction:\n{instr}\n\n### Response:\n"

def ensure_plain_prompt(text: str) -> str:
    """当没有 instruction 字段时，用已有 prompt；若缺少 Response 前缀则补上"""
    text = (text or "").strip()
    if "### Response:" not in text:
        text = f"{text}\n\n### Response:\n"
    return text

def _clean(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.replace("\r", " ").strip()
    if not s or s.upper() in {"[EMPTY_RESPONSE]", "[EMPTY_THOUGHT]"}:
        return ""
    s = re.sub(r"[ \t]+", " ", s)
    return s

def load_preference_dataset(path: str) -> Dataset:
    rows: List[Dict[str, str]] = []
    total, dropped = 0, 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                j = json.loads(line)
            except Exception:
                dropped += 1
                continue

            instr = _clean(j.get("instruction") or "")
            # prompt 字段仅在没有 instruction 时回退
            raw_prompt = _clean(j.get("prompt") or "")

            ch = j.get("chosen", {})
            rj = j.get("rejected", {})
            chosen   = _clean(ch.get("response") if isinstance(ch, dict) else (ch or ""))
            rejected = _clean(rj.get("response") if isinstance(rj, dict) else (rj or ""))

            if not (chosen and rejected):
                dropped += 1
                continue

            if instr:
                prompt = build_plain_prompt_from_instruction(instr)
            else:
                if not raw_prompt:
                    dropped += 1
                    continue
                prompt = ensure_plain_prompt(raw_prompt)

            rows.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})

    print(f"[Data] total={total} | valid={len(rows)} | dropped={dropped}")
    if not rows:
        raise RuntimeError("No valid DPO pairs after filtering.")
    return Dataset.from_list(rows)


def main():
    # 设备与随机种子
    torch.cuda.set_device(0)
    set_seed(GLOBAL_SEED)
    random.seed(GLOBAL_SEED)

    # 分词器
    tok = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH, trust_remote_code=True, use_fast=True
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    # 数据
    full_ds = load_preference_dataset(DPO_DATA_PATH)
    split = full_ds.train_test_split(test_size=0.1, seed=GLOBAL_SEED)
    train_ds, eval_ds = split["train"], split["test"]
    print(f"[Split] Train={len(train_ds)} | Eval={len(eval_ds)}")

    # Base + LoRA
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    base.config.use_cache = False  # 训练省显存
    if hasattr(base, "gradient_checkpointing_enable"):
        base.gradient_checkpointing_enable()

    # 挂 LoRA（可训练）
    peft_sig = inspect.signature(PeftModel.from_pretrained)
    if "is_trainable" in peft_sig.parameters:
        model = PeftModel.from_pretrained(base, ADAPTER_PATH, is_trainable=True)
    else:
        model = PeftModel.from_pretrained(base, ADAPTER_PATH, inference_mode=False)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"[LoRA] Trainable params: {trainable:,} / {total:,}")
    if trainable == 0:
        raise RuntimeError("LoRA not trainable. Check peft version or adapter path.")

    # 参考模型 / rDPO
    ref = None
    if not USE_REFERENCE_FREE:
        ref = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        ).eval()
        if hasattr(ref, "gradient_checkpointing_disable"):
            ref.gradient_checkpointing_disable()
        ref.config.use_cache = True

    # DPO 配置
    cfg = DPOConfig(
        output_dir=DPO_OUTPUT_DIR,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=4e-6,
        lr_scheduler_type="cosine",
        warmup_steps=50,
        max_steps=1000,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        eval_strategy="steps",   # 注意参数名
        eval_steps=50,
        bf16=True,
        beta=0.1,
        max_prompt_length=512,
        max_completion_length=512,
        truncation_mode="keep_end",
        padding_value=tok.pad_token_id,
        report_to="wandb",
        run_name="dpo-lora-llama3-plainfmt",
        logging_dir="./wandb_logs",
        remove_unused_columns=False,
        reference_free=USE_REFERENCE_FREE,
        save_safetensors=True,
    )

    # Trainer（兼容不同 TRL 版本）
    trl_sig = inspect.signature(DPOTrainer)
    if "processing_class" in trl_sig.parameters:
        trainer = DPOTrainer(
            model=model, ref_model=ref, args=cfg,
            train_dataset=train_ds, eval_dataset=eval_ds,
            processing_class=tok,
        )
    else:
        trainer = DPOTrainer(model, ref, cfg, tok, train_ds, eval_ds)

    # 训练
    trainer.train()

    # 仅保存 LoRA 适配器
    final_adapter_dir = os.path.join(DPO_OUTPUT_DIR, "adapter-final")
    model.save_pretrained(final_adapter_dir)

    print(f"[Training] Finished. Checkpoints at {DPO_OUTPUT_DIR}")
    print(f"[Save] PEFT adapter saved to: {final_adapter_dir}")


if __name__ == "__main__":
    main()
