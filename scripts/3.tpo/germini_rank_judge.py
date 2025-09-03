#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import json
import time
import requests
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

# ===== Path configuration =====
INPUT_PATH = Path("/home/zceexl3/ai_accountant/scripts/3.tpo/output/thought_response_20250830_104536.jsonl")
OUTPUT_PATH = Path("/home/zceexl3/ai_accountant/scripts/3.tpo/output/preference_pairs_gemini.jsonl")

# ===== Gemini configuration (recommend using environment variables) =====
API_KEY   = "AIzaSyBEie1_V2cXwBrOUe07C3R-gnTPWDhI6nc"
MODEL_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"

# ===== HTTP basics =====
TIMEOUT_S = 30
MAX_RETRY = 4
BACKOFF0  = 0.8

session = requests.Session()

# ===== Cleaning functions =====
CTRL_CHARS = "".join(map(chr, list(range(0,32)) + [127]))
CTRL_TABLE = str.maketrans("", "", CTRL_CHARS)

def clean_text(s: str) -> str:
    if not s:
        return s
    s = s.translate(CTRL_TABLE)
    s = re.sub(r"\bassistant\b", "", s, flags=re.IGNORECASE)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = s.strip()
    return s

# ===== Key: determine whether text is substantive (replaces is_nonempty) =====
PLACEHOLDER_RE = re.compile(
    r"""^\s*(?:\[
            (?:EMPTY_RESPONSE|EMPTY_THOUGHT)
        \]
        |N/?A
        |NONE?
        |NULL
        |QUizzysystemquiz
        |\.\.\.
    )\s*$""",
    re.IGNORECASE | re.VERBOSE
)
ONLY_PUNCT_RE = re.compile(r"^[\W_]+$", re.UNICODE)

def is_substantive_text(s: str) -> bool:
    """Treat placeholders, pure punctuation/underscores, very short strings, and null/NA as invalid."""
    if not s:
        return False
    s = s.strip()
    if not s:
        return False
    if PLACEHOLDER_RE.match(s):
        return False
    if ONLY_PUNCT_RE.match(s):
        return False
    # Require at least 4 alphanumeric characters to avoid noise like "ok", "—", "."
    if len(re.findall(r"[A-Za-z0-9]", s)) < 4:
        return False
    return True

# ===== Evaluation prompt (STRICT JSON) =====
def build_judge_prompt(instruction: str, response: str) -> str:
    return f"""
You are an impartial UK tax expert and evaluator.
Evaluate ONLY the **Response** for helpfulness, UK tax accuracy (HMRC/legislation-compliant), completeness, and clarity. Ignore any hidden chain-of-thought.

Return a STRICT JSON object with exactly these keys:
- "score": integer from 0 to 100 (0=unusable, 50=partly helpful with notable issues, 100=excellent and fully correct).
- "reason": one concise sentence.

Do NOT add extra text. Do NOT include markdown fences.

Instruction:
{instruction}

Response:
{response}
""".strip()

def safe_parse_json(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    i, j = text.find("{"), text.rfind("}")
    if i != -1 and j != -1 and j > i:
        text = text[i:j+1]
    try:
        return json.loads(text)
    except Exception:
        return None

def call_gemini_json(prompt: str):
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.0}
    }
    for attempt in range(1, MAX_RETRY + 1):
        try:
            resp = session.post(MODEL_URL, json=payload, timeout=TIMEOUT_S)
            if resp.status_code == 200:
                data = resp.json()
                try:
                    text = data["candidates"][0]["content"]["parts"][0]["text"]
                except Exception:
                    raise ValueError(f"Unexpected schema: {data}")
                obj = safe_parse_json(text)
                if not obj or "score" not in obj or "reason" not in obj:
                    raise ValueError(f"Non-JSON or missing keys: {text[:200]}")
                score = int(obj["score"])
                score = max(0, min(100, score))
                return {"score": score, "reason": str(obj["reason"]).strip()}
            elif resp.status_code in (408, 429, 500, 502, 503, 504):
                time.sleep(BACKOFF0 * (2 ** (attempt - 1)))
                continue
            else:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        except requests.RequestException:
            time.sleep(BACKOFF0 * (2 ** (attempt - 1)))
            continue
        except Exception:
            time.sleep(BACKOFF0 * (2 ** (attempt - 1)))
            continue
    return None

# ===== Load & group =====
def load_candidates(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def group_by_instruction(items):
    groups = defaultdict(list)
    for it in items:
        pid = str(it.get("pair_id", ""))
        instr_id = pid.split("_")[0] if "_" in pid else pid
        it["instruction_id"] = instr_id
        it["instruction"] = clean_text(it.get("instruction", ""))
        it["response"]    = clean_text(it.get("response", ""))
        it["thought"]     = clean_text(it.get("thought", ""))
        groups[instr_id].append(it)
    return groups

def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

# ===== Main pipeline =====
def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    items = list(load_candidates(INPUT_PATH))
    if not items:
        print("Input is empty.")
        return

    groups = group_by_instruction(items)

    skipped_insufficient = 0
    dropped_placeholders = 0
    written = 0

    # Overwrite the output file to avoid appending stale data
    with open(OUTPUT_PATH, "w", encoding="utf-8") as out_f:
        for instr_id, cand_list in tqdm(groups.items(), desc="Evaluating groups"):
            if not cand_list:
                continue
            instruction = cand_list[0].get("instruction", "")
            if not instruction:
                continue

            # Filter out non-substantive responses first
            filtered = []
            for c in cand_list:
                r = c.get("response", "")
                if is_substantive_text(r):
                    filtered.append(c)
                else:
                    dropped_placeholders += 1
            if len(filtered) < 2:
                skipped_insufficient += 1
                continue

            # Score with Gemini
            scored = []
            for cand in filtered:
                resp_text = cand.get("response", "")
                prompt = build_judge_prompt(instruction, resp_text)
                res = call_gemini_json(prompt)
                if res is not None:
                    scored.append({
                        "score":  res["score"],
                        "reason": res["reason"],
                        "item":   cand
                    })

            # Deduplicate by response text
            seen = set()
            deduped = []
            for x in sorted(scored, key=lambda z: z["score"], reverse=True):
                rt = norm(x["item"].get("response", ""))
                if not rt or rt in seen:
                    continue
                seen.add(rt)
                deduped.append(x)

            if len(deduped) < 2:
                skipped_insufficient += 1
                continue

            best = deduped[0]
            best_resp = norm(best["item"].get("response", ""))

            # From the low end, find the first "valid and different from best" as the worst
            worst = None
            for x in reversed(deduped):
                rt = norm(x["item"].get("response", ""))
                if rt and rt != best_resp and is_substantive_text(rt):
                    worst = x
                    break

            if worst is None:
                skipped_insufficient += 1
                continue

            out_obj = {
                "instruction": instruction,
                "instruction_id": instr_id,
                "chosen": {
                    "response": best["item"].get("response", ""),
                    "thought":  best["item"].get("thought", ""),
                    "score":    best["score"],
                    "reason":   best["reason"],
                    "meta": {
                        "pair_id":     best["item"].get("pair_id", ""),
                        "temperature": best["item"].get("temperature"),
                        "top_p":       best["item"].get("top_p"),
                        "seed_hint":   best["item"].get("seed_hint"),
                        "attn_impl":   best["item"].get("attn_impl"),
                    }
                },
                "rejected": {
                    "response": worst["item"].get("response", ""),
                    "thought":  worst["item"].get("thought", ""),
                    "score":    worst["score"],
                    "reason":   worst["reason"],
                    "meta": {
                        "pair_id":     worst["item"].get("pair_id", ""),
                        "temperature": worst["item"].get("temperature"),
                        "top_p":       worst["item"].get("top_p"),
                        "seed_hint":   worst["item"].get("seed_hint"),
                        "attn_impl":   worst["item"].get("attn_impl"),
                    }
                }
            }

            out_f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            out_f.flush()
            written += 1

    print(f"[DONE] Preference pairs written to {OUTPUT_PATH}")
    print(f"[INFO] Written pairs: {written}")
    print(f"[INFO] Dropped placeholders/non-substantive: {dropped_placeholders}")
    print(f"[INFO] Skipped groups (insufficient valid & distinct): {skipped_insufficient}")

if __name__ == "__main__":
    main()
