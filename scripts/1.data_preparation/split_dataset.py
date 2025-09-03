# scripts/split_dataset.py

import json
from pathlib import Path
from sklearn.model_selection import train_test_split

# === Input dataset path ===
DATASET_PATH = Path("/home/zceexl3/ai_accountant/data/uk_tax_synthetic_dataset.jsonl")

# === Output paths ===
OUTPUT_DIR = DATASET_PATH.parent
TRAIN_PATH = OUTPUT_DIR / "train.jsonl"
VAL_PATH   = OUTPUT_DIR / "val.jsonl"
TEST_PATH  = OUTPUT_DIR / "test.jsonl"

# === Load data ===
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    records = [json.loads(line) for line in f]

# === First split: train (80%) + temp (20%) ===
train_set, temp_set = train_test_split(records, test_size=0.2, random_state=42)

# === Second split: validation (10%) + test (10%) from temp ===
val_set, test_set = train_test_split(temp_set, test_size=0.5, random_state=42)

# === Write train set ===
with open(TRAIN_PATH, "w", encoding="utf-8") as f:
    for item in train_set:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

# === Write validation set ===
with open(VAL_PATH, "w", encoding="utf-8") as f:
    for item in val_set:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

# === Write test set ===
with open(TEST_PATH, "w", encoding="utf-8") as f:
    for item in test_set:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

# === Summary ===
print("✅ Dataset split completed!")
print(f"Training set   : {len(train_set)} records -> {TRAIN_PATH}")
print(f"Validation set : {len(val_set)} records   -> {VAL_PATH}")
print(f"Test set       : {len(test_set)} records  -> {TEST_PATH}")
