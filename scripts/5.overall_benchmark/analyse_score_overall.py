#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

# ========= Paths =========
OUT_DIR = Path("/home/zceexl3/ai_accountant/scripts/overall_benchmark")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_SCORES = {
    "base": OUT_DIR / "base_scores.jsonl",
    "sft":  OUT_DIR / "sft_scores.jsonl",
    "tpo":  OUT_DIR / "tpo_scores.jsonl",
    "tpo2": OUT_DIR / "tpo2_scores.jsonl",
}

# ========= Outputs =========
SUMMARY_CSV = OUT_DIR / "score_summary.csv"
WIDE_CSV    = OUT_DIR / "scores_wide.csv"
PLOT_SCORE  = OUT_DIR / "score_dist_box_clean"   # -> .png
PLOT_ECDF   = OUT_DIR / "score_ecdf"             # -> .png
PLOT_SURV   = OUT_DIR / "score_survival"         # -> .png

# ========= Helpers =========
def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows

def coerce_score_1_100(obj: Dict[str, Any]):
    """Try multiple fields/patterns to coerce a score into [1, 100] integer."""
    s = obj.get("judge_score", None)
    if isinstance(s, (int, float)):
        si = int(s)
        if 1 <= si <= 100:
            return si
    if isinstance(s, str) and s.strip().isdigit():
        si = int(s.strip())
        if 1 <= si <= 100:
            return si
    raw = str(obj.get("judge_raw", ""))
    m = re.search(r"\b(100|[1-9]?\d)\b", raw)
    if m:
        val = int(m.group(1))
        if 1 <= val <= 100:
            return val
    return None

def load_scores_from_file(path: Path, label: str) -> pd.DataFrame:
    """Load scores from a jsonl file into a tidy DataFrame."""
    rows = []
    for j in read_jsonl(path):
        score = coerce_score_1_100(j)
        if score is None:
            continue
        rows.append({"model": label, "instruction": j.get("instruction", ""), "score": score})
    return pd.DataFrame(rows)

def discover_score_files() -> Dict[str, Path]:
    """Find expected files + any '*_scores.jsonl' inside OUT_DIR."""
    found = {}
    for label, p in EXPECTED_SCORES.items():
        if p.exists():
            found[label] = p
    for p in OUT_DIR.glob("*_scores.jsonl"):
        name = p.stem
        label = name.replace("_scores", "")
        if label not in found:
            found[label] = p
    return found

def preferred_order(labels: List[str]) -> List[str]:
    base_order = ["base", "sft", "tpo", "tpo2"]
    labels_set = set(labels)
    ordered = [x for x in base_order if x in labels_set]
    others = sorted([x for x in labels if x not in base_order])
    return ordered + others

def save_png(fig: plt.Figure, basepath: Path, dpi: int = 300):
    fig.tight_layout()
    fig.savefig(f"{basepath}.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"📊 Saved: {basepath}.png")

def ecdf_array(y: np.ndarray):
    """Return x (unique sorted) and ECDF values for array y."""
    y = np.sort(np.asarray(y))
    x_unique, counts = np.unique(y, return_counts=True)
    cum = np.cumsum(counts) / len(y)
    return x_unique, cum

# ========= Load =========
label_to_file = discover_score_files()
if not label_to_file:
    raise SystemExit(f"No '*_scores.jsonl' files found in {OUT_DIR}.")

frames = []
for label, path in label_to_file.items():
    part = load_scores_from_file(path, label)
    if part.empty:
        print(f"⚠️  {label}: {path} has no valid scores. Skipping.")
        continue
    frames.append(part)
if not frames:
    raise SystemExit("Failed to parse valid scores from any file.")

df = pd.concat(frames, ignore_index=True)

# ========= Summary Tables =========
models_in_df = preferred_order(df["model"].unique().tolist())
df["model"] = pd.Categorical(df["model"], categories=models_in_df, ordered=True)

summary = df.groupby("model")["score"].agg(["count", "mean", "std", "min", "max"]).round(2)
print("\n=== Score Summary (1–100) ===")
print(summary)
summary.to_csv(SUMMARY_CSV)
print(f"✅ Saved summary: {SUMMARY_CSV}")

wide = df.pivot_table(index="instruction", columns="model", values="score", aggfunc="first")
wide.to_csv(WIDE_CSV)
print(f"✅ Saved wide table: {WIDE_CSV}")

# ========= Visualization =========
sns.set(style="whitegrid", font_scale=1.15)

# ---- A) Box plot (show μ and median inside the box) ----
fig = plt.figure(figsize=(max(9, 2.0 * len(models_in_df) + 5), 6))
ax = plt.gca()

# Boxplot
sns.boxplot(
    x="model", y="score", data=df, width=0.5, showcaps=True, showfliers=False,
    boxprops=dict(alpha=0.35), whiskerprops=dict(alpha=0.85),
    medianprops=dict(color="black", lw=2)  # black line = median
)

# Stats
g = df.groupby("model")["score"]
means   = g.mean().round(1)
medians = g.median().round(1)
q1      = g.quantile(0.25)
q3      = g.quantile(0.75)

# Put μ and median text inside the box
for i, m in enumerate(models_in_df):
    if m in means.index:
        y_text = q3[m] - 0.05 * (q3[m] - q1[m]) if q3[m] > q1[m] else (q1[m] + q3[m]) / 2
        ax.text(
            i, y_text,
            f"μ={means[m]:.1f}\nmed={medians[m]:.1f}",
            ha="center", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.75", alpha=0.95)
        )

# Overall mean (annotated box at bottom-right, not in legend)
overall_mean = df["score"].mean()
ax.axhline(overall_mean, linestyle="--", linewidth=1, color="blue")
ax.text(
    0.985, 0.07, f"Overall mean = {overall_mean:.1f}",
    transform=ax.transAxes, ha="right", va="bottom",
    bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.6", alpha=0.98)
)

# Legend explaining Median once
median_handle = Line2D([0], [0], color="black", lw=2, label="Median (black line)")
ax.legend(handles=[median_handle], loc="upper right", frameon=True)

ax.set_title("Judge Score Distribution (1–100) by Model")
ax.set_ylabel("Judge Score (1–100)")
ax.set_xlabel("Model Variant")
ax.set_ylim(1, 100)
save_png(fig, PLOT_SCORE)

# ---- B) ECDF ----
fig = plt.figure(figsize=(9, 5.2))
ax = plt.gca()
for mdl in models_in_df:
    sub = df.loc[df["model"] == mdl, "score"].to_numpy()
    if len(sub) == 0:
        continue
    x_e, y_e = ecdf_array(sub)
    ax.step(x_e, y_e, where="post", label=mdl, linewidth=2, alpha=0.95)
ax.set_title("ECDF of Judge Scores")
ax.set_xlabel("Judge Score")
ax.set_ylabel("Cumulative Probability")
ax.set_xlim(1, 100)
ax.set_ylim(0, 1.0)
ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
ax.legend(title="Model")
save_png(fig, PLOT_ECDF)

# ---- C) Survival curve (1 - ECDF) ----
fig = plt.figure(figsize=(9, 5.2))
ax = plt.gca()
thresholds = [60, 70, 80]
legend_extra = []
for mdl in models_in_df:
    sub = df.loc[df["model"] == mdl, "score"].to_numpy()
    if len(sub) == 0:
        continue
    x_e, y_e = ecdf_array(sub)
    surv_y = 1 - y_e
    ax.step(x_e, surv_y, where="post", label=mdl, linewidth=2, alpha=0.95)
    legend_extra.append((mdl, (sub >= 60).mean()))

for t in thresholds:
    ax.axvline(t, color="0.5", lw=1, ls="--")

ax.set_title("Survival Curve (1 - ECDF): P(Score > x)")
ax.set_xlabel("Judge Score Threshold x")
ax.set_ylabel("Probability (Score > x)")
ax.set_xlim(1, 100)
ax.set_ylim(0, 1.0)
ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

# Append P≥60 in legend labels
handles, labels = ax.get_legend_handles_labels()
labels_fmt = []
for lbl in labels:
    p = next((p for m, p in legend_extra if m == lbl), None)
    labels_fmt.append(f"{lbl}  (P≥60={p:.2f})" if p is not None else lbl)
ax.legend(handles, labels_fmt, title="Model")
save_png(fig, PLOT_SURV)

print("\n✅ Done.")
