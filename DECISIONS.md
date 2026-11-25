# DECISIONS

## 2026-11-25 Environment substitutions (Prompt 0)
- Native Windows 11 + PowerShell: `just` replaces `make` (CLAUDE.md "make X" == `just X`). `just check` = lint + test.
- No Docker: MLflow runs natively (`just mlflow`, sqlite backend, `./mlartifacts`); no docker-compose.yml, Dockerfile or Terraform; llama.cpp baseline skipped.
- Dependency `mlflow` added now (section 3 lists MLflow; needed by tracking.py). Dev: pytest, ruff, mypy.
- CI uses `just` via extractions/setup-just.

## 2026-11-25 Spike outcome (Prompt 1.1)
- Default model `Qwen/Qwen3-0.6B` (rev c1899de289a04d12100db370d81485cdf75e47ca). Qwen3-1.7B not attempted (CPU-only machine).
- Export: onnxruntime-genai builder, FP32 CPU. Optimum 2.3.0 has no ONNX exporter, so the `optimum` dependency was removed.
- Mixed precision: two-pass `MatMulNBitsQuantizer` (4-bit with the other nodes excluded, then 8-bit). `quantize_dynamic` is not used (it yields MatMulInteger and no INT4). See docs/SPIKE.md.
- BFCL checker is not importable without a heavy dependency tree; write the minimal matcher. The `bfcl-eval` wheel is the data source only (not a project dependency; fetched to a local path).
- Extra dependencies beyond section 3, kept for the code: `onnx` (graph inspection), `onnx-ir` (genai builder), `accelerate` (transformers loading), `matplotlib` (report charts).
