# CLAUDE.md: ToolQuant

Save this file as `CLAUDE.md` in the repo root. Read it fully at the start of every session.

## 1. What this project is

ToolQuant searches per-layer-group precision for a small tool-calling language model using **tool-call success** as the objective, builds the result as a real ONNX Runtime model, and compares it with uniform INT8, uniform INT4, a perplexity-guided mixed map and llama.cpp `Q4_K_M`.

Hypothesis H1: at the same model size, a precision map chosen by tool-call sensitivity gives higher tool-call success than a map chosen by perplexity. **Report the result in whichever direction it falls, with confidence intervals.** A clean negative result is an acceptable outcome.

Non-goals: a new quantization algorithm, training from scratch, GPU serving, supporting many model families (one in Phase 1, a second in Phase 2).

## 2. Rules for you (Claude Code)

1. Never invent a number. Every metric in README, docs or commits comes from a file in `results/` or an MLflow run written by code in this repo.
2. Do not add a dependency outside section 3 without recording the reason in `DECISIONS.md` first.
3. List a technology in the README only if it is used in the code.
4. Test first for the scorer, fake-quantizer, size model, allocator and statistics.
5. If the spec is ambiguous or an API differs from what is written here, check the installed package's docs or source, record the decision in `DECISIONS.md`, and continue.
6. Run `make check` before declaring a task done and paste the last 20 lines of its output.
7. Small commits. No secrets, no model weights, no large files in Git (section 11).
8. Fixed seeds everywhere. Log the seed, the git SHA, the model revision and package versions in every MLflow run.
9. Probe data and validation data never overlap. A test enforces it.

## 3. Stack

| Part | Choice |
|---|---|
| Language | Python 3.12 |
| Model and probing | PyTorch, Hugging Face `transformers`, `datasets` |
| Default model | `Qwen/Qwen3-0.6B` (Qwen3-1.7B if the spike shows it exports and fits); the spike decides |
| Export and runtime | ONNX Runtime, its quantization tools (`MatMulNBitsQuantizer`, `quantize_dynamic`), and `onnxruntime-genai` or Hugging Face Optimum for export; the spike decides |
| Evaluation data | BFCL tool-calling tasks (public Gorilla repository data); reuse the BFCL AST checker if it can be imported, else implement the minimal matcher in section 6 |
| Statistics | numpy, scipy, statsmodels (McNemar) |
| Tracking | MLflow (local server in Docker Compose) |
| Serving (Phase 2) | FastAPI, uvicorn, ONNX Runtime, OpenTelemetry |
| Baseline | llama.cpp with a `Q4_K_M` GGUF of the same model (Phase 2) |
| Packaging | Docker, GitHub Actions |
| Cloud (Phase 2) | GCP Cloud Run (CPU) for the serving image |

## 4. Repo layout

```
toolquant/
  CLAUDE.md  PROMPTS.md  DECISIONS.md  OPEN_QUESTIONS.md  README.md  Makefile
  docker-compose.yml  .env.example  pyproject.toml  uv.lock
  src/toolquant/
    data/ (bfcl.py, wikitext.py, splits.py)
    scoring/ (parse.py, match.py, taxonomy.py, metrics.py)
    quant/ (fakequant.py, groups.py, sizing.py, probe.py, allocate.py, interactions.py)
    export/ (onnx_export.py, apply_map.py, parity.py)
    evalrun/ (generate.py, run_eval.py, stats.py)
    bench/ (speed.py, memory.py)
    serve/ (app.py, otel.py)
    tracking.py
  maps/            # committed precision maps (small JSON)
  results/         # small JSON and CSV outputs, committed per release
  tests/  docs/  docker/
```

## 5. Model, groups, quantization, sizes

**Layer groups.** Default granularity is one group per decoder block (28 groups for Qwen3-0.6B and Qwen3-1.7B). An option `--granularity attn_mlp` splits each block into attention and MLP (56 groups). The embedding table and `lm_head` stay at FP16 (or INT8 in the ONNX build) and are not searched. Group definitions come from the model config and module names, never hard-coded indices.

**Fake quantization** (`quant/fakequant.py`): weight-only, per-group symmetric, round-to-nearest. For weight `W` with input dimension grouped by `group_size` (default 128): `scale = max(|w|) / (2^(bits-1) - 1)` per group, `q = clamp(round(w / scale), -2^(bits-1), 2^(bits-1) - 1)`, `W_hat = q * scale`. Support `bits` in 4 and 8. Applied in place on a copy of the model's linear layers for the chosen group.

**Size model** (`quant/sizing.py`): `size_bits(group) = params(group) * (bits + 16 / group_size)` (an FP16 scale per weight group). Total includes unquantized parameters at 16 bits. Budgets are expressed as a fraction between all-INT4 size and all-INT8 size: `budget(f) = size_int4 + f * (size_int8 - size_int4)`, for f in `{0.25, 0.5}` by default. Validate the size model against the actual ONNX file size and record the difference.

## 6. Scoring

Prompting: use the model's chat template with `tools` (via `tokenizer.apply_chat_template`). Greedy decoding, `max_new_tokens = 256`, stop at end-of-turn. Parse tool calls from the model's native format (for Qwen3, JSON inside `<tool_call>` tags); verify the exact format from the tokenizer's chat template, not from memory.

Per-prompt record: `{id, category, needs_call, calls_made, predicted, reference, outcomes}` with these definitions.

| Metric | Definition |
|---|---|
| `schema_valid` | Output parses as the model's tool-call format and validates against the tool's JSON schema |
| `ast_match` | Function name equals the reference and every required argument equals the reference after type normalisation (ints and floats, string case rules as in BFCL); extra optional arguments allowed |
| `missed_call` | `needs_call` is true and no call was made |
| `silent_failure` | `schema_valid` is true, a call was made, and `ast_match` is false (well-formed but wrong function or arguments) |
| `abstain_ok` | `needs_call` is false and no call was made |
| `tool_call_success` | Reported success: `ast_match` for call-needed prompts, `abstain_ok` for no-call prompts |

If the BFCL checker can be imported, call it for `ast_match` and record any divergence from the definition above in `DECISIONS.md`. Add a taxonomy wrapper on top; do not fork the checker.

Splits (`data/splits.py`): fixed seed; `probe` = 100 prompts (balanced across categories), `validation` = all remaining prompts for the chosen categories, `multi_turn` (Phase 2) separate. The split file `data/splits.json` is committed.

## 7. Algorithm

1. **Baseline.** FP16 model scored on probe and validation; log with bootstrap intervals.
2. **Probe** (`quant/probe.py`): for each group `g`, fake-quantize only `g` to 4-bit, all others stay FP16, score the 100 probe prompts. `sensitivity_g = baseline_success - success_g`. Also compute perplexity change on a fixed WikiText-2 slice (64 sequences of 512 tokens) for the same single-group perturbation. Save both tables.
3. **Allocate** (`quant/allocate.py`): sort groups by ascending sensitivity; assign 4-bit to the least sensitive until the size budget is reached, the rest INT8. Produce two maps per budget: `toolcall` (sensitivity) and `ppl` (perplexity change). Map file: `maps/<model>_<budget>_<kind>.json` with `{model, revision, group_size, granularity, bits_per_group, size_bits, seed, git_sha}`.
4. **Interactions** (Phase 2, `quant/interactions.py`): for the k least sensitive groups, compare the joint score drop with the sum of single drops; if the joint drop exceeds the sum, run swap local search (swap one 4-bit group with one INT8 group; accept if the probe score improves; fixed iteration cap). Confirm any change on validation, never select on validation.
5. **Export** (`export/apply_map.py`): build FP ONNX once; apply the map with ONNX Runtime quantization: INT4 `MatMulNBits` for 4-bit groups and INT8 dynamic quantization for the others. The week-1 spike must determine, on the installed version, how to apply different precisions to different MatMul nodes (for example `nodes_to_exclude` on the 4-bit quantizer then INT8 on the excluded set). Write the result to `docs/SPIKE.md`.
6. **Parity** (`export/parity.py`): on 200 prompts, compare greedy next-token agreement between the fake-quantized PyTorch model and the real ONNX model for the same map. Report agreement rate and first-divergence position.
7. **Validate** (`evalrun/`): score every configuration on `validation`. Configurations: FP16, uniform INT8, uniform INT4, `ppl` map and `toolcall` map at each budget, and (Phase 2) llama.cpp `Q4_K_M`.
8. **Statistics** (`evalrun/stats.py`): paired bootstrap, 10,000 resamples, over per-prompt binary outcomes; 95% interval of the difference in `tool_call_success`, `missed_call`, `silent_failure`. McNemar exact test for paired binary outcomes. Report both. Never report a difference without its interval.
9. **Benchmark** (`bench/`): on a stated machine, fixed prompt length (128 tokens) and output length (128 tokens), 5 warm-up and 20 timed runs; record decode tokens per second, time to first token, p95 latency, peak resident memory; also the CPU model and thread count.

## 8. Tracking

MLflow experiment `toolquant`. One run per `(model, config)`. Params: model, revision, group size, granularity, budget, map kind, seed, git SHA. Metrics: the six scoring metrics with interval bounds, size MB, tokens per second, TTFT, p95 latency, peak memory, parity agreement. Artifacts: the map JSON, per-prompt record file, the ONNX model path or hash (the file itself is not in Git).

## 9. Serving (Phase 2)

FastAPI: `POST /v1/generate` with `{messages, tools}` returns the parsed tool call and raw text; `GET /healthz` runs one tool-call prompt and expects a valid call. OpenTelemetry span `generate` with attributes `prompt_tokens`, `output_tokens`, `ttft_ms`, `map_id`. ONNX Runtime session options (threads, graph optimisation level, IO binding) are set from environment variables.

## 10. Commands (Makefile)

`make env` · `make probe` · `make allocate` · `make export MAP=...` · `make parity` · `make eval` · `make bench` · `make serve` · `make report` (README tables and charts from `results/`) · `make lint` · `make test` · `make check` (lint, tests, smoke eval on 50 prompts of the tiny test config).

## 11. Artifacts and Git

In Git: code, `maps/*.json`, `data/splits.json`, small results files, MLflow run exports. Not in Git: ONNX models, GGUF files, datasets. Models go to the Hugging Face Hub or a bucket; check the base model's licence before publishing weights.

## 12. Definition of done

**Phase 1:** FP16 baseline with intervals; sensitivity tables (tool-call and perplexity) for all groups; real ONNX models for uniform INT8, uniform INT4, `ppl` map and `toolcall` map; parity agreement reported; a results table with size, tokens per second and tool-call success with intervals for each configuration; `make check` green.

**Phase 2:** interaction test and swap search; multi-turn tasks; llama.cpp baseline; at least 10 configurations compared; Docker image serving the best map; OpenTelemetry spans; CI smoke eval; a second model compared; write-up with a clearly stated conclusion.

## 13. Metrics to record (code writes them)

Groups probed; probe and validation prompt counts; configurations compared; size in MB and ratio to FP16; the six scoring metrics per configuration with intervals; differences versus the `ppl` map with intervals and McNemar p-values; parity agreement; tokens per second, TTFT, p95 latency, peak memory with CPU model and thread count; MLflow run count; CI smoke eval time.

## 14. Resume rules

Claim "built" and "measured". Never "first" or "novel". State the hardware next to every speed number. Do not call the CPU result "on-device". If the map does not beat the perplexity map, say so in the README and write the resume bullet about the measurement method and the finding.

> **Note 2026-11-25 (spike result):** CLAUDE.md section 7 step 5 said INT8 dynamic quantization for non-4-bit groups. On ORT 1.32.0 the working route is two `MatMulNBitsQuantizer` passes (4-bit, then 8-bit with block size 128); `quantize_dynamic` is not used. The genai builder produces 4 MatMul nodes per decoder layer. BFCL AST checker is not importable; the minimal matcher is used. See docs/SPIKE.md.
