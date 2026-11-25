# ToolQuant

ToolQuant searches per-layer-group precision for a small tool-calling language model using tool-call success as the objective, builds the result as an ONNX Runtime model, and compares it with uniform INT8, uniform INT4 and a perplexity-guided mixed map.

## Problem

## Method

## Results (generated)

## Reproduce

**Fetch BFCL data** (public, from the `bfcl-eval` wheel; not a project dependency):

```powershell
uv run python scripts/fetch_bfcl.py   # writes data/cache/bfcl
```
## Limits
