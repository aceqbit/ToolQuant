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

## 2026-11-25 Scoring and data (Prompt 1.2)
- Categories: `simple_python` (400), `multiple` (200), `irrelevance` (240) = 840 prompts; probe 34/33/33, validation 366/167/207. Parallel and live categories left out to keep CPU evaluation of ~740 prompts x 6+ configs feasible. Irrelevance is the only `needs_call=false` category.
- Matcher differences from BFCL: (a) optional arguments the model adds that are absent from the reference are allowed (CLAUDE.md section 6); BFCL would fail them; optional arguments that appear in the reference must match. (b) Strings compare lowercase with space and `,./-_*^` removed. (c) Function names compared exactly (BFCL dotted names kept). (d) bool never equals a number.
- A malformed call attempt (bad JSON or truncated `<tool_call>`) counts as a call made (`calls_made`>0): not `missed_call`, not `silent_failure` (not schema-valid), not success. Label `malformed_or_invalid`.
- Rate denominators: success, missed_call, silent_failure over all prompts (so paired per-prompt differences use the same population); ast_match over call-needed; abstain_ok over no-call; schema_valid over prompts with a call.
- Oracle check on all 840 tasks (reference fed back as output): 839 pass; `simple_python_200` fails (BFCL reference quirk), kept in the data.
- BFCL data comes from the `bfcl-eval` wheel via `scripts/fetch_bfcl.py`, not a local Gorilla checkout.

## 2026-11-26 Phase 2 dependencies
- fastapi, uvicorn, opentelemetry-sdk, opentelemetry-exporter-otlp-proto-http: serving and spans (CLAUDE.md section 3, Phase 2). Already present transitively via mlflow; now declared explicitly.
- No Docker, Dockerfile, Terraform or llama.cpp baseline (environment substitutions); serving runs natively with uvicorn.
