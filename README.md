# specdecode

**Lookahead Logit Acceleration** — a zero-cost speculative decoding scheme that reuses discarded forward-pass logits as draft tokens for the next step.

## The Idea

Every autoregressive forward pass produces logits at **all** positions, but only the last one is used. Due to the causal mask, `logits[i]` attends only to tokens `0..i` — identical to what a fresh forward pass at position `i` would produce. These are exact draft tokens, not approximations.

```
Step t:   forward([t1..tn])
          → accept  tn+1 = argmax(logits[n])     ← standard AR
          → free draft tn+2 = argmax(logits[n+1])
          → free draft tn+3 = argmax(logits[n+2])
          ...

Step t+1: forward([t1..tn+1])
          → verify draft for tn+2, accept if match
          → KV cache: pointer rewind only, no recompute
```

Why acceptance rates are high:
- Causal mask → free logits are mathematically exact, not approximate
- Subword tokenization → 2–4 tokens per word; completions are near-certain once a word starts
- High local mutual information in natural language
- Instruct/RLHF fine-tuning → lower entropy, more deterministic continuations

## Results (Qwen2.5-0.5B-Instruct)

| Text type        | Match rate | Expected free tokens/step |
|------------------|-----------|--------------------------|
| quicksort code   | 97.6%     | 40.0                     |
| fibonacci code   | 80.6%     | 4.2                      |
| structured list  | 63.9%     | 1.8                      |
| ML explanation   | 61.9%     | 1.6                      |
| transformer text | 41.3%     | 0.7                      |
| **Aggregate**    | **69.1%** | **~2.2**                 |

~2.2 expected free tokens per step → **~3x theoretical speedup**, on the smallest tested model. Larger instruct models are expected to score higher on their own output.

## Files

| File | Purpose |
|------|---------|
| `validate_lookahead_v3.py` | Quick validation on 5 fixed texts |
| `mt_bench_lookahead.py` | Full MT-Bench benchmark (CLI) |
| `mt_bench_lookahead_colab_v2.ipynb` | Colab notebook (G4/H100 optimized) |
| `CONTEXT.md` | Full research context and conversation history |

## Quick Start

```bash
pip install transformers accelerate torch

# Validate on small model (~2 min, CPU-OK)
python validate_lookahead_v3.py

# MT-Bench on 7B (needs GPU)
python mt_bench_lookahead.py --model Qwen/Qwen2.5-7B-Instruct

# MT-Bench full run on 32B (G4/H100 recommended, ~30 min)
python mt_bench_lookahead.py --model Qwen/Qwen2.5-32B-Instruct --max-questions 65
```

## Relation to Prior Work

| Method | Draft source | Training required | Distribution exact? |
|--------|-------------|-------------------|---------------------|
| **This work** | Same model's lookahead logits | No | Yes (causal mask) |
| EAGLE / Medusa | Trained draft head on hidden states | Yes | No |
| Lookahead Decoding | Jacobi iteration | No | No (convergence-dependent) |
| Blockwise Parallel (Stern 2018) | Auxiliary networks | Yes | No |

## Next Steps

1. Run full MT-Bench on 32B for publishable numbers vs EAGLE-3
2. Implement the scheme in HuggingFace `generate()` or vLLM
3. Compare wall-clock speedup vs theoretical speedup
4. Write up as blog post / preprint

## How It Differs from Jacobi Decoding

Jacobi decoding requires multiple passes to converge and invalidates the KV cache on each iteration because token values change. This scheme uses the **current** pass's logits as drafts for the **next** pass — no convergence needed, no cache invalidation. KV cache management is just a pointer rewind (confirmed in vLLM: `request.py_rewind_len`).
