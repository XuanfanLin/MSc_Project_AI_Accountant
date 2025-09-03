import os
import json
import numpy as np
from pathlib import Path
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==== Paths ====
DATA_FILE = Path("/home/zceexl3/ai_accountant/data/uk_tax_synthetic_dataset.jsonl")
EDA_DIR = Path("/home/zceexl3/ai_accountant/scripts/1.data_preparation/eda")
EDA_DIR.mkdir(parents=True, exist_ok=True)

# 图形风格与全局字号
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "figure.dpi": 120,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

# 可调参数 —— 控制条形图密度
TOP_K_FIRST_WORDS = 15          # 仅展示前 K 个首词
MIN_FREQ_FIRST_WORD = 5         # 频次低于该阈值的忽略（可设为 1 关闭）
CLIP_PCT = 99                   # 长度分布按百分位裁剪尾部，便于阅读（如 99 或 99.5）

# ==== Load Full Dataset ====
with open(DATA_FILE, "r", encoding="utf-8") as f:
    records = [json.loads(line) for line in f]
df = pd.DataFrame(records)

# Ensure required columns exist
if "instruction" not in df.columns or "output" not in df.columns:
    raise ValueError("Dataset must contain 'instruction' and 'output' fields.")

# Create optional 'input' column if missing
if "input" not in df.columns:
    df["input"] = ""

# ==== Add Length Features (character counts) ====
df["instruction_len"] = df["instruction"].fillna("").astype(str).str.len()
df["input_len"]       = df["input"].fillna("").astype(str).str.len()
df["output_len"]      = df["output"].fillna("").astype(str).str.len()

# ==== Save Stats & Print Summary ====
print(f"\n📊 DATASET - {len(df)} samples")
stats = df[["instruction_len", "input_len", "output_len"]].describe()
stats.to_csv(EDA_DIR / "all_length_stats.csv", index=True)
print(stats)

# ==== First-Word table (with filtering & top-k) ====
first_words = (
    df["instruction"].fillna("")
      .astype(str).str.strip()
      .str.split().str[0].str.lower()
)

# 频次统计与过滤
fw_counts = first_words.value_counts()
fw_counts = fw_counts[fw_counts >= MIN_FREQ_FIRST_WORD]  # 过滤低频
total_covered = int(fw_counts.head(TOP_K_FIRST_WORDS).sum())
coverage = total_covered / max(1, len(df)) * 100

first_words_df = fw_counts.head(TOP_K_FIRST_WORDS).rename_axis("first_word").reset_index(name="frequency")
first_words_df.to_csv(EDA_DIR / "all_first_words_topk.csv", index=False)

print("\nTop first words (filtered):")
print(first_words_df.head(10))

# ==== Plot 1: Instruction vs. Output Length Distribution ====
# 使用共享分箱（Freedman–Diaconis）+ 裁剪长尾，让直方图更清晰
instr = df["instruction_len"].to_numpy()
outp  = df["output_len"].to_numpy()

# 长尾裁剪
i_clip = np.percentile(instr, CLIP_PCT)
o_clip = np.percentile(outp,  CLIP_PCT)
clip_max = max(i_clip, o_clip)
instr_clipped = np.clip(instr, a_min=0, a_max=clip_max)
outp_clipped  = np.clip(outp,  a_min=0, a_max=clip_max)

# 共享分箱（fd 更稳健；若数据很少可改为 'auto'）
bins = np.histogram_bin_edges(np.concatenate([instr_clipped, outp_clipped]), bins='fd')

plt.figure(figsize=(10.5, 5.2))
sns.histplot(instr_clipped, bins=bins, kde=True, stat="count", alpha=0.45, label="Instruction")
sns.histplot(outp_clipped,  bins=bins, kde=True, stat="count", alpha=0.45, label="Output")
plt.title(f"Instruction vs. Output Length Distribution (clipped @ P{CLIP_PCT})")
plt.xlabel("Text Length (characters)")
plt.ylabel("Count")
plt.legend()
plt.tight_layout()
plt.savefig(EDA_DIR / "length_distribution_instruction_vs_output.png", dpi=300)
plt.close()

# ==== Plot 2: Top-K First Words (Horizontal Bar, sorted) ====
# 横向条形图 + 频次标注，避免密集挤在一起
plt.figure(figsize=(10.5, 6.0))
sorted_df = first_words_df.sort_values("frequency", ascending=True)
ax = sns.barplot(data=sorted_df, y="first_word", x="frequency")
plt.title(f"Top-{len(sorted_df)} Leading Words in Instructions "
          f"(≥{MIN_FREQ_FIRST_WORD} freq, covers {coverage:.1f}% of samples)")
plt.xlabel("Frequency")
plt.ylabel("First word")
# 在条形图上标注频次
for p in ax.patches:
    w = p.get_width()
    y = p.get_y() + p.get_height()/2
    ax.text(w + max(sorted_df["frequency"])*0.01, y, f"{int(w)}", va="center", ha="left", fontsize=9)

plt.tight_layout()
plt.savefig(EDA_DIR / "top_first_words_horizontal.png", dpi=300, bbox_inches="tight")
plt.close()

print(f"\n✅ All EDA results saved in: {EDA_DIR}")
print(f"   - Stats CSV: {EDA_DIR / 'all_length_stats.csv'}")
print(f"   - First words CSV: {EDA_DIR / 'all_first_words_topk.csv'}")
print(f"   - Length plot: {EDA_DIR / 'length_distribution_instruction_vs_output.png'}")
print(f"   - First words plot: {EDA_DIR / 'top_first_words_horizontal.png'}")
