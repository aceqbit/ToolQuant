# ToolQuant

ToolQuant searches per-layer-group precision for a small tool-calling language model using tool-call success as the objective, builds the result as an ONNX Runtime model, and compares it with uniform INT8, uniform INT4 and a perplexity-guided mixed map.

## Problem

Weight quantization is usually judged by perplexity. For an agent-style model the question that matters is whether it still emits the right function call. Hypothesis H1: at the same model size, a precision map chosen by tool-call sensitivity gives higher tool-call success than a map chosen by perplexity. The result is reported in whichever direction it falls, with confidence intervals; a negative result is an acceptable outcome.

## Method

- Model: `Qwen/Qwen3-0.6B`, 28 decoder blocks as 28 groups (embedding and `lm_head` are not searched).
- Data: BFCL v4 single-turn tasks (`simple_python`, `multiple`, `irrelevance`; 840 prompts). A fixed-seed split gives 100 probe prompts and 740 validation prompts with no overlap (a test enforces it).
- Scoring: greedy decoding with the Qwen3 chat template and tools; six metrics (schema validity, AST match, missed call, silent failure, abstain, tool-call success) from a minimal matcher (the BFCL checker is not importable without heavy dependencies; see `docs/SPIKE.md`, `DECISIONS.md`).
- Probe: fake-quantize one group at a time to 4-bit (weight-only, per-group of 128, symmetric round-to-nearest), score the probe prompts, and record the drop in tool-call success. The perplexity change on a fixed WikiText-2 slice is recorded for the same perturbation.
- Allocation: lowest-sensitivity groups get 4-bit, the rest 8-bit, up to a size budget between all-INT4 and all-INT8 (`toolcall` and `ppl` maps at the same budget).
- Build: ONNX Runtime `MatMulNBits` with 4-bit and 8-bit nodes in one model (two quantizer passes, see `docs/SPIKE.md`).
- Statistics: paired bootstrap (10,000 resamples) and exact McNemar over per-prompt outcomes.

## Results (generated)

Run `just report` after the evaluations in `results/` exist.

## Reproduce

Native Windows with PowerShell, `uv` and `just`; Python 3.12.

```powershell
just env                                   # uv sync
uv run python scripts/fetch_bfcl.py        # BFCL data -> data/cache/bfcl (from the public bfcl-eval wheel)
uv run python -m toolquant.data.splits     # regenerates data/splits.json (committed)
just test                                  # lint + tests: just check
```

GPU steps (FP16 baseline and the sensitivity probe) run on a free Kaggle or Colab GPU: see `docs/GPU.md`.
Then, on CPU:

```powershell
just allocate results/sensitivity_qwen3-0.6b_<utc>.json   # writes maps/*.json
just export int8; just export int4; just export maps/qwen3-0.6b_0.25_toolcall.json
just parity maps/qwen3-0.6b_0.25_toolcall.json
just eval int8 int4 qwen3-0.6b_0.25_toolcall qwen3-0.6b_0.25_ppl
just bench int4
just report
just mlflow                                 # serves mlflow.db on http://127.0.0.1:5000
```

## Limits

- One model (Qwen3-0.6B), three BFCL categories, 740 validation prompts; intervals are wide for small differences.
- BFCL may be in the model's training data, so only same-model comparisons are meaningful.
- The ONNX build keeps the embedding table and `lm_head` in FP32 and the quantization scales in FP32; sizes in the results are file sizes of that build.
- Speeds are CPU numbers on one laptop; they are not on-device measurements.
- The matcher is a minimal one and differs from the BFCL checker in documented ways (`DECISIONS.md`).
