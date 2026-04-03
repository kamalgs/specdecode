# Lookahead Logit Acceleration — Project Context

## What This Is

A novel inference optimization hypothesis being validated experimentally.
This document captures the full context of an ongoing research conversation
so it can be continued in Claude Code or a new session.

---

## The Core Idea

**Hypothesis:** Standard autoregressive LLM inference throws away useful information.
Every forward pass produces logits at ALL token positions, but only the last one is used.
The remaining logits are free draft token predictions — and due to the causal mask,
they are computed under *exactly* the same conditions as a true autoregressive pass.

**The causal mask argument:**
During training, causal masking ensures `logits[i]` attends only to tokens `0..i`.
At inference, this means `logits[i+1]` from a forward pass ending at position `n`
is *identical* to `logits[i+1]` from a fresh forward pass starting at `i+1`.
There is no distribution mismatch — these are exact, not approximate, draft tokens.

**The proposal:**
```
Step t:   Forward pass on [t1..tn]
          → Confirm token tn+1 = greedy(logits[n])       (standard AR)
          → FREE draft k=1: greedy(logits[n+1])           (predicts tn+2)
          → FREE draft k=2: greedy(logits[n+2])           (predicts tn+3)
          ...

Step t+1: Forward pass on [t1..tn+1]
          → Confirms tn+2, checks if free draft for tn+2 was correct
          → Generates next set of free drafts
          → KV cache just gets a pointer rewind for rejected tokens (no recompute)
```

**Why acceptance rate should be high:**
1. **Causal mask** → free logits are exact, not approximate
2. **Subword tokens** → a single "word" is 2-4 tokens; once committed, continuations are near-certain
3. **Word2vec / local MI** → natural language has high local mutual information; each token strongly constrains the next 3-4
4. **Instruct/RLHF training** → lower entropy outputs, more deterministic continuations
5. **Sliding window attention effectiveness** → local context is sufficient for most predictions; many heads only attend to the last few tokens

**Why this hasn't been published:**
The field jumped to trained draft heads (EAGLE) and Jacobi methods without exploiting
this simpler "free logit reuse" approach. The causal mask + subword tokenization
insight isn't prominently assembled in the literature.

**Closest existing work:**
- Lookahead Decoding (LMSYS) — Jacobi-based, different mechanism
- EAGLE/Medusa — use main model hidden states but require training
- Bonus token optimization in vLLM — one token only, not sliding window
- Blockwise Parallel Decoding (Stern et al. 2018) — early precursor

---

## Experimental Validation

### What we measured

For each token position `i` in a generated response:
- `draft_prob`: P(actual token | logits[i]) — the free lookahead logit's confidence
- `greedy match`: does `argmax(logits[i]) == actual_token[i+1]`?
- Run-length: how many consecutive correct predictions (= consecutive free tokens)
- Expected free tokens per step = mean run length

### Results so far (Qwen2.5-0.5B-Instruct, fixed text)

| Text type        | Match rate | Expected free tokens |
|------------------|-----------|---------------------|
| quicksort code   | 97.6%     | 40.0                |
| fibonacci code   | 80.6%     | 4.2                 |
| structured list  | 63.9%     | 1.8                 |
| ML explanation   | 61.9%     | 1.6                 |
| transformer text | 41.3%     | 0.7                 |
| **Aggregate**    | **69.1%** | **~2.2**            |

Token type breakdown:
- punct: 85–100% match
- subword continuations: 75–100% match
- space_word (word-initial): 37–98% match (varies by domain)

**Key insight:** Even a tiny 0.5B model shows ~2.2 expected free tokens per step,
implying ~3x theoretical speedup. A 7B+ instruct model on its own output would be higher.

**Note on measurement:** The v1 script had an off-by-one bug (comparing same logit twice).
v3 is correct: measures `greedy(logits[i]) == actual_token[i+1]`.

---

## Files

### Scripts
- `validate_lookahead_v3.py` — standalone Python script, correct measurement
- `mt_bench_lookahead.py` — full MT-Bench evaluation script (CLI)
- `mt_bench_lookahead_colab_v2.ipynb` — Colab notebook, G4/H100 optimized

### Data
- `lookahead_results_v2.json` — raw results from 0.5B model on fixed text

---

## Colab Notebook Details

**File:** `mt_bench_lookahead_colab_v2.ipynb`

**12 cells:**
1. Install deps (`transformers`, `accelerate`, `flash-attn`)
2. Imports + GPU check
3. Config form: `RUN_MODE` (quick_test | full), `MODEL_SIZE`, `MAX_NEW_TOKENS`
4. MT-Bench questions (65 questions, 8 categories, embedded)
5. Load model
6. **Quick test cell** — single coding question, token-by-token match table, ~2 min
7. Core analysis function
8. Run benchmark (live per-question output)
9. Summary table + comparison to EAGLE-3/draft model published numbers
10. 4-panel visualization (match by category, free tokens, run-length dist, token type)
11. Save + download JSON + PNG

**GPU recommendations:**
- G4 (L40S 48GB) → Qwen2.5-32B-Instruct, bfloat16, flash_attention_2
- H100 (80GB) → same, ~2x faster
- A100 (40GB) → 14B
- T4 (16GB) → 1.5B

**Model map:**
```python
'0.5B': 'Qwen/Qwen2.5-0.5B-Instruct'
'1.5B': 'Qwen/Qwen2.5-1.5B-Instruct'
'7B':   'Qwen/Qwen2.5-7B-Instruct'
'14B':  'Qwen/Qwen2.5-14B-Instruct'
'32B':  'Qwen/Qwen2.5-32B-Instruct'
```

**Recommended flow:**
1. Runtime → Change runtime type → G4 GPU
2. Run cells 1→6 (quick test) — verify ✅ in ~2 min
3. Change `RUN_MODE = 'full'` in cell 3
4. Re-run cells 3→11 — full benchmark ~30 min
5. Cell 12 auto-downloads `lookahead_32B_full.json` + `lookahead_results.png`

---

## Key Technical Details

### Why KV cache is not "polluted"
The causal mask means `logits[i]` was computed attending only to `0..i`.
When a draft token is rejected, the KV cache just needs a pointer rewind
(set length back to n+m where m accepted tokens). The prefix entries are
already valid — no recomputation needed. This is confirmed in vLLM source:
`request.py_rewind_len = num_draft_tokens_allocated - num_accepted_tokens`.

### Why this differs from Jacobi decoding
Jacobi requires multiple passes to converge and has KV cache invalidation
because tokens change between iterations. This proposal uses the current pass's
logits as drafts for the *next* pass — no convergence needed, no invalidation.

### Sampling / temperature handling
Same rejection sampling scheme as standard speculative decoding (Leviathan et al.):
accept token with probability min(1, p(x)/q(x)) where p = main model, q = draft.
Here q = p (same model), so acceptance simplifies: accept if greedy matches,
or apply the correction distribution. Output distribution is preserved exactly.

---

## Next Steps / Open Questions

1. **Run full MT-Bench on 32B** — get publishable numbers comparable to EAGLE-3
2. **Measure on model's own generated text** vs fixed reference text (current scripts do generated text correctly)
3. **Implement actual sliding window** — instead of just measuring, implement the scheme in a fork of vLLM or HuggingFace generate()
4. **Compare to EAGLE-3** on same hardware/model — is the zero-cost scheme competitive?
5. **Check if this is already in any inference engine** — closest found was vLLM bonus token issue (#4212) but not the full sliding window approach
6. **Paper/blog post** — the four-argument case (causal mask + subword + local MI + instruct training) is clean and novel enough

---

## How to Continue in Claude Code

```bash
# Install deps
pip install transformers accelerate torch

# Quick validation on small model
python validate_lookahead_v3.py

# Full MT-Bench run (needs GPU)
python mt_bench_lookahead.py --model Qwen/Qwen2.5-7B-Instruct --max-questions 65

# Or open the Colab notebook
# mt_bench_lookahead_colab_v2.ipynb
```

When continuing this conversation, share this file and say:
"Continue the lookahead logit acceleration project. Context is in CONTEXT.md."

---

## Conversation Summary

The conversation developed the idea organically:

1. Started with speculative decoding mechanics (verification pass, KV cache)
2. Identified that verification logits are computed for all positions simultaneously
3. Proposed: why not use the same single model's lookahead logits as draft tokens?
4. Identified this as related to but distinct from Jacobi/Lookahead decoding
5. Confirmed KV cache "pollution" is just pointer management, not correctness issue
6. Built convergent argument: causal mask + subword + word2vec local MI + instruct training + sliding window attention
7. Searched literature — no paper found doing exactly this
8. Built validation scripts iteratively (v1 had off-by-one bug, v3 is correct)
9. Got first results: 69% aggregate match on 0.5B, 97% on code
10. Extended to MT-Bench for publishable comparison to EAGLE-3 numbers
11. Built Colab notebook optimized for G4/H100 with quick-test cell
