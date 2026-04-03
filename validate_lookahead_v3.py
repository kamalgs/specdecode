"""
validate_lookahead_v3.py -- Lookahead Logit Acceleration validation script

Measures how often the "free" lookahead logits from a single forward pass
correctly predict the next token, across several text domains.

Key measurement (v3, fixed from v1 off-by-one):
  For each position i in [0, seq_len-2]:
    draft_token = argmax(logits[i])          # prediction made at position i
    actual_token = input_ids[i+1]            # true next token
    match = (draft_token == actual_token)

A match means we could have gotten this token for free in the lookahead scheme.
"""

import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import defaultdict

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

# Fixed reference texts for reproducible comparison
TEXTS = {
    "quicksort_code": """\
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

arr = [3, 6, 8, 10, 1, 2, 1]
print(quicksort(arr))
""",
    "fibonacci_code": """\
def fibonacci(n):
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]
    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[-1] + fib[-2])
    return fib

print(fibonacci(10))
""",
    "structured_list": """\
Top 5 programming languages in 2024:
1. Python - widely used for data science, ML, and web development
2. JavaScript - essential for web development, both frontend and backend
3. TypeScript - typed superset of JavaScript, growing in popularity
4. Rust - systems programming with memory safety guarantees
5. Go - simple, fast, great for cloud-native development
""",
    "ml_explanation": """\
Transformer models revolutionized natural language processing by introducing
the self-attention mechanism. Unlike RNNs, transformers process all tokens
in parallel, enabling much faster training on modern hardware. The attention
mechanism allows each token to directly attend to all other tokens in the
sequence, capturing long-range dependencies without the vanishing gradient
problem that plagued earlier architectures.
""",
    "transformer_text": """\
The attention mechanism computes query, key, and value projections for each
token. The attention weights are computed as the softmax of the scaled dot
product between queries and keys. These weights determine how much each
value contributes to the output representation. Multi-head attention runs
this process in parallel with different learned projections, allowing the
model to jointly attend to information from different representation subspaces.
""",
}


def classify_token(token_str: str) -> str:
    """Classify a token into a type for breakdown analysis."""
    if not token_str.strip():
        return "whitespace"
    if all(c in ".,;:!?()[]{}\"'-" for c in token_str.strip()):
        return "punct"
    if token_str.startswith(" ") or token_str.startswith("\u0120"):
        return "space_word"
    return "subword_cont"


def analyze_text(model, tokenizer, text: str, device: str) -> dict:
    """
    Run a single forward pass and measure lookahead match rate.

    For each position i, logits[i] is the model's prediction for token i+1.
    We measure how often argmax(logits[i]) == actual token at position i+1.
    """
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"][0]  # (seq_len,)
    seq_len = input_ids.shape[0]

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0]  # (seq_len, vocab_size)

    # logits[i] predicts position i+1; compare to input_ids[i+1]
    draft_tokens = logits[:-1].argmax(dim=-1)  # (seq_len-1,)
    actual_tokens = input_ids[1:]               # (seq_len-1,)
    matches = (draft_tokens == actual_tokens).cpu().numpy()

    # Draft probabilities (the model's confidence in the actual next token)
    probs = torch.softmax(logits[:-1], dim=-1)
    actual_tokens_cpu = actual_tokens.cpu()
    draft_probs = probs[torch.arange(seq_len - 1), actual_tokens_cpu].float().numpy()

    actual_tokens_np = actual_tokens_cpu.numpy()

    # Compute run lengths starting at each position
    run_lengths = np.zeros(len(matches), dtype=int)
    i = len(matches) - 1
    while i >= 0:
        if matches[i]:
            run_lengths[i] = 1 + (run_lengths[i + 1] if i + 1 < len(matches) else 0)
        i -= 1

    # Token type breakdown
    type_matches: dict = defaultdict(list)
    for i in range(len(matches)):
        token_str = tokenizer.decode([int(actual_tokens_np[i])])
        tok_type = classify_token(token_str)
        type_matches[tok_type].append(bool(matches[i]))

    type_stats = {
        k: {"match_rate": float(np.mean(v)), "count": len(v)}
        for k, v in type_matches.items()
    }

    match_rate = float(matches.mean())
    # Expected free tokens per step (weighted by match probability)
    expected_free = float(
        (run_lengths * matches).sum() / max(1, matches.sum())
    )

    unique, counts = np.unique(run_lengths, return_counts=True)
    run_dist = {str(int(k)): int(v) for k, v in zip(unique, counts)}

    return {
        "seq_len": int(seq_len),
        "match_rate": match_rate,
        "expected_free_tokens": expected_free,
        "mean_draft_prob": float(draft_probs.mean()),
        "token_type_stats": type_stats,
        "run_length_dist": run_dist,
    }


def print_token_table(model, tokenizer, text: str, device: str, max_rows: int = 30):
    """Print a token-by-token match table for a short text."""
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"][0]

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0]

    draft_tokens = logits[:-1].argmax(dim=-1)
    actual_tokens = input_ids[1:]
    probs = torch.softmax(logits[:-1], dim=-1)

    print(f"\n{'i':>4}  {'actual':>12}  {'draft':>12}  {'prob':>7}  {'match'}")
    print("-" * 55)
    for i in range(min(max_rows, len(actual_tokens))):
        actual = tokenizer.decode([int(actual_tokens[i])])
        draft = tokenizer.decode([int(draft_tokens[i])])
        prob = float(probs[i, actual_tokens[i]])
        match = "OK" if draft_tokens[i] == actual_tokens[i] else "--"
        print(f"{i:>4}  {repr(actual):>12}  {repr(draft):>12}  {prob:7.4f}  {match}")


def main():
    print(f"Loading {MODEL_ID} ...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map=device,
    )
    model.eval()

    results = {}
    print("=" * 62)
    print(f"{'Text':25s}  {'Match%':>8s}  {'E[free]':>8s}  {'P(draft)':>9s}")
    print("=" * 62)

    for name, text in TEXTS.items():
        stats = analyze_text(model, tokenizer, text, device)
        results[name] = stats
        print(
            f"{name:25s}  {stats['match_rate']*100:7.1f}%  "
            f"{stats['expected_free_tokens']:8.1f}  "
            f"{stats['mean_draft_prob']:9.4f}"
        )

    all_match_rates = [r["match_rate"] for r in results.values()]
    all_free = [r["expected_free_tokens"] for r in results.values()]
    print("-" * 62)
    print(
        f"{'Aggregate':25s}  {np.mean(all_match_rates)*100:7.1f}%  "
        f"{np.mean(all_free):8.1f}"
    )

    # Token type breakdown (aggregated across all texts)
    print("\nToken type breakdown (aggregate):")
    combined: dict = defaultdict(list)
    for r in results.values():
        for tok_type, s in r["token_type_stats"].items():
            combined[tok_type].extend([s["match_rate"]] * s["count"])
    for tok_type, rates in sorted(combined.items()):
        print(f"  {tok_type:20s}: {np.mean(rates)*100:.1f}%  (n={len(rates)})")

    output_path = "lookahead_results_v3.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
