import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# fast_extract_code_5k.py — Fast, CPU-multithreaded, batched, robust

import os
# Tune threads for Ryzen 7 5700X (8C/16T). Try 12-16 if you have headroom.
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["OMP_NUM_THREADS"] = "12"
os.environ["MKL_NUM_THREADS"] = "12"

import json
from pathlib import Path
from transformers import AutoTokenizer
from tqdm import tqdm
import random
import re

# Config
DATASET_FILE = Path(os.path.join(BASE_DIR, "Section-B/am_0.9M.jsonl")
OUTPUT_FILE  = Path(os.path.join(BASE_DIR, "Section-B/deepseek_thinking_5k_code_512.jsonl")
BASE_MODEL   = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
TARGET = 5000
BATCH_SIZE = 512   # tokenization batch size; reduce to 256 if RAM-bound

random.seed(42)

# Heuristics
CODE_KEYWORDS_USER = [
    "implement", "write a function", "write code", "algorithm", "optimize",
    "time complexity", "space complexity", "python", "javascript", "java",
    "pseudocode", "bug", "debug", "fix the code", "refactor", "unit test",
    "constraints", "edge case", "api", "class", "function", "return", "array",
    "string", "tree", "graph", "dynamic programming", "greedy", "binary search",
    "dfs", "bfs", "two pointers", "sliding window",
]
MATH_EXCLUDES = [
    "derivative", "integral", "theorem", "lemma", "proof", "algebra",
    "trigonometry", "geometry", "matrix", "polynomial", "equation",
]
CODE_PATTERNS_ASSISTANT = [
    re.compile(r"```"),
    re.compile(r"\bdef\s+\w+\s*$$"),    # Python def
    re.compile(r"\bclass\s+\w+\s*:"),   # Python class
    re.compile(r"\breturn\b"),
    re.compile(r"\bfor\s+\w+\s+in\b"),
    re.compile(r"\bwhile\s*$$"),
    re.compile(r"\bimport\s+\w+"),
    re.compile(r"\bfrom\s+\w+\s+import\b"),
]

def quick_code_score(user_msg: str, assistant_msg: str) -> int:
    u = user_msg.lower()
    a = assistant_msg
    score = 0
    if any(mx in u for mx in MATH_EXCLUDES):
        score -= 2
    if any(kw in u for kw in CODE_KEYWORDS_USER):
        score += 2
    if any(p.search(a) for p in CODE_PATTERNS_ASSISTANT):
        score += 3
    return score

def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except:
                continue

def build_text(user_msg, assistant_msg):
    # Minimal text for token length estimation
    return f"System: You are a coding assistant.\nUser: {user_msg}\nAssistant: {assistant_msg}"

def main():
    print("⚡ FAST CODE-HEAVY EXTRACTION — 5k")
    print("CPU-parallel tokenizer, batched, early-filtered")
    print("="*60)

    if not DATASET_FILE.exists():
        print(f"❌ Dataset not found: {DATASET_FILE}")
        return

    print("🔧 Loading tokenizer (CPU only)...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    print("✅ Tokenizer ready")

    selected = []
    candidates_texts = []     # texts waiting for batch tokenization
    candidates_payloads = []  # (user_msg, assistant_msg, score)

    # Adaptive thresholds
    accept_threshold = 2      # start strict: favor clearer code signals
    min_fill_relax_at = 1500  # if we don't hit 1500 after scanning 200k, relax
    scanned = 0

    pbar = tqdm(iter_jsonl(DATASET_FILE), desc="Scanning", unit="lines")
    for ex in pbar:
        if len(selected) >= TARGET:
            break
        scanned += 1

        msgs = ex.get("messages")
        if not msgs or len(msgs) < 2:
            continue

        user_msg, assistant_msg = "", ""
        for m in msgs:
            r = m.get("role", "")
            c = m.get("content", "")
            if r == "user":
                user_msg = c
            elif r == "assistant":
                assistant_msg = c

        if not user_msg or not assistant_msg or len(assistant_msg) < 50:
            continue

        score = quick_code_score(user_msg, assistant_msg)
        if score <= -2:
            continue

        # Queue for batch tokenization
        candidates_texts.append(build_text(user_msg, assistant_msg))
        candidates_payloads.append((user_msg, assistant_msg, score))

        # Periodically relax if not filling fast enough
        if scanned % 200_000 == 0:
            if len(selected) < min_fill_relax_at:
                accept_threshold = max(1, accept_threshold - 1)  # relax
                print(f"⚠️ Relaxing acceptance threshold to {accept_threshold} after {scanned:,} lines")

        # Tokenize in batch
        if len(candidates_texts) >= BATCH_SIZE:
            enc = tokenizer(
                candidates_texts,
                return_tensors="np",
                truncation=False,
                add_special_tokens=True,
            )
            # Reliable length from shape
            lengths = [len(ids) for ids in enc["input_ids"]]

            for (u, a, s), tok_len in zip(candidates_payloads, lengths):
                if tok_len <= 512:
                    # Accept if strong signal, or if still early (bootstrap)
                    if s >= accept_threshold or len(selected) < 1000:
                        selected.append({
                            "system": "You are a helpful assistant for coding tasks.",
                            "conversations": [
                                {"role": "user", "value": u},
                                {"role": "assistant", "value": a},
                            ]
                        })
                        if len(selected) >= TARGET:
                            break

            candidates_texts.clear()
            candidates_payloads.clear()

    # Flush remaining
    if len(selected) < TARGET and candidates_texts:
        enc = tokenizer(
            candidates_texts,
            return_tensors="np",
            truncation=False,
            add_special_tokens=True,
        )
        lengths = [len(ids) for ids in enc["input_ids"]]
        for (u, a, s), tok_len in zip(candidates_payloads, lengths):
            if tok_len <= 512:
                if s >= accept_threshold or len(selected) < 1000:
                    selected.append({
                        "system": "You are a helpful assistant for coding tasks.",
                        "conversations": [
                            {"role": "user", "value": u},
                            {"role": "assistant", "value": a},
                        ]
                    })
                    if len(selected) >= TARGET:
                        break

    print(f"\n📊 SUMMARY")
    print(f"  Selected: {len(selected)} / {TARGET}")
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    print(f"💾 Saving to {OUTPUT_FILE}")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for ex in selected:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print("✅ Done")

if __name__ == "__main__":
    main()
