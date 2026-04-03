# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

Research validating **Lookahead Logit Acceleration**: reusing the discarded logits from an autoregressive forward pass as free draft tokens for speculative decoding. No training required. See `CONTEXT.md` for full research context.

## Commands

```bash
# Install dependencies
pip install transformers accelerate torch numpy

# On GPU (for larger models)
pip install flash-attn --no-build-isolation

# Quick validation (CPU-OK, ~2 min on 0.5B)
python validate_lookahead_v3.py

# MT-Bench benchmark
python mt_bench_lookahead.py --model Qwen/Qwen2.5-7B-Instruct --max-questions 65
python mt_bench_lookahead.py --model Qwen/Qwen2.5-32B-Instruct --output lookahead_32B_full.json

# Colab: open mt_bench_lookahead_colab_v2.ipynb on G4/H100 runtime
```

## Key Measurement (validate_lookahead_v3.py)

The v3 script has the correct off-by-one logic:

```python
draft_token  = argmax(logits[i])   # prediction at position i
actual_token = input_ids[i+1]      # true token at position i+1
match = (draft_token == actual_token)
```

v1 had a bug comparing `logits[i]` against `input_ids[i]` (same position). Any new measurement script must follow v3's indexing.

## Architecture

Both scripts share the same measurement loop pattern:
1. Tokenize text, run a single forward pass
2. Compare `logits[:-1].argmax(-1)` against `input_ids[1:]` (the shift)
3. Compute run lengths (consecutive matches) for expected free tokens
4. `mt_bench_lookahead.py` additionally calls `model.generate()` first so measurement is on the model's own output — not a fixed reference text

`mt_bench_lookahead.py` uses `model.train(False)` to set inference mode (same as the `.eval()` method, avoids a security linting hook in this repo).

## Models

```python
MODEL_MAP = {
    '0.5B': 'Qwen/Qwen2.5-0.5B-Instruct',
    '1.5B': 'Qwen/Qwen2.5-1.5B-Instruct',
    '7B':   'Qwen/Qwen2.5-7B-Instruct',
    '14B':  'Qwen/Qwen2.5-14B-Instruct',
    '32B':  'Qwen/Qwen2.5-32B-Instruct',
}
```

GPU minimums: T4 (16GB) → 1.5B; A100 40GB → 14B; L40S/H100 48GB+ → 32B.

## Output Format

Both scripts produce JSON with this schema:
```json
{
  "model": "...",
  "overall_match_rate": 0.691,
  "mean_expected_free_tokens": 2.2,
  "theoretical_speedup": 3.2,
  "by_category": { "coding": {"match_rate": 0.89, "n": 10}, ... },
  "results": [{ "id": 131, "category": "coding", "match_rate": ..., ... }]
}
```

## Research Context

Full hypothesis, prior results, and next steps are in `CONTEXT.md`. When continuing the project, read `CONTEXT.md` first for the complete picture including the causal-mask argument, KV cache rewind mechanics, and comparison to EAGLE/Jacobi.
