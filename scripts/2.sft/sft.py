#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import Trainer  # duplicate import kept as in original

# ===== Environment Setup =====
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ===== Custom Trainer =====
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items() if torch.is_tensor(v)}
        outputs = model(**inputs)
        return (outputs.loss, outputs) if return_outputs else outputs.loss

# ===== Format Function =====
def format_example(ex):
    instr = ex["instruction"]
    inp = ex.get("input", "")
    out = ex["output"]

    if inp:
        text = (
            f"### Instruction:\n{instr}\n\n"
            f"### Input:\n{inp}\n\n"
            f"### Response:\n{out}"
        )
    else:
        text = (
            f"### Instruction:\n{instr}\n\n"
            f"### Response:\n{out}"
        )

    tok = tokenizer(text, truncation=True, padding="max_length", max_length=512)
    tok["labels"] = tok["input_ids"].copy()
    return tok


def main():
    MODEL_PATH = "/home/zceexl3/ai_accountant/models/LLM-Research/Meta-Llama-3-8B"
    OUTPUT_DIR = "/home/zceexl3/ai_accountant/models/lora-sft"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ===== Tokenizer =====
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    # ===== Load Base Model with 4bit Quantization =====
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True
    )

    # Enable gradient checkpointing
    base_model = prepare_model_for_kbit_training(
        base_model,
        use_gradient_checkpointing=True
    )

    # ===== LoRA Config =====
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=0.05,  # lower dropout for small datasets
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    # ===== Load Datasets =====
    data_files = {
        "train": "/home/zceexl3/ai_accountant/data/train.jsonl",
        "validation": "/home/zceexl3/ai_accountant/data/val.jsonl"
    }
    raw_datasets = load_dataset("json", data_files=data_files)

    train_dataset = raw_datasets["train"].map(
        format_example,
        remove_columns=raw_datasets["train"].column_names
    )
    eval_dataset = raw_datasets["validation"].map(
        format_example,
        remove_columns=raw_datasets["validation"].column_names
    )

    # ===== Training Arguments =====
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=5e-5,          # lower LR
        warmup_ratio=0.1,
        weight_decay=0.01,
        bf16=True,
        eval_strategy="steps",       # kept as-is per original
        eval_steps=10,               # evaluate every 10 steps
        save_strategy="steps",
        save_steps=10,               # save every 10 steps
        save_total_limit=2,          # keep at most 2 checkpoints
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_strategy="steps",
        logging_steps=10,
        logging_first_step=True,
        log_level="info",
        remove_unused_columns=False,
        report_to="wandb",
        run_name="lora-sft-v1",
        logging_dir="./wandb_logs",
    )

    # ===== Trainer =====
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    # ===== Train =====
    trainer.train()
    model.save_pretrained(OUTPUT_DIR)

    # ===== Inference Sample =====
    model.eval()
    prompt = (
        "### Instruction:\n"
        "Calculate the personal allowance for UK tax year 2024/25.\n\n"
        "### Response:\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out_ids = model.generate(
        **inputs,
        max_new_tokens=128,
        temperature=0.7,
        use_cache=False
    )
    gen = out_ids[0][inputs["input_ids"].shape[1]:]
    print("Generated:\n", tokenizer.decode(gen, skip_special_tokens=True))


if __name__ == "__main__":
    main()
