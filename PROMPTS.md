# PROMPTS.md: ToolQuant build prompts for Claude Code

Paste the prompts in order. One prompt per session: run `/clear` first, work on a branch named after the prompt (`p1-4-probe`), commit, merge when the acceptance checks pass.

Before prompt 0: create an empty repo, copy `CLAUDE.md` (from `toolquant_CLAUDE.md`) into its root, and install Docker, Python 3.12 and `uv`. Heavy steps (the probe) run on a free Kaggle or Colab GPU notebook; every prompt tells you which machine to use. Claude Code writes the scripts; you run the GPU ones and bring the result files back into `results/`.

Each prompt has: **Read**, **Do**, **Constraints**, **Accept** (commands that must pass), **Report**. If an accept check fails, fix it before reporting. Report in the format given, with no extra prose.

---

## Prompt 0: bootstrap

**Read:** `CLAUDE.md` in full.

**Do:**
1. Create the layout from section 4 with empty modules, a `uv` project with `pytest`, `ruff`, `mypy`, and the Makefile targets from section 10 (stubs exit 1 except `lint` and `test`).
2. `docker-compose.yml` with an MLflow server (local file store or sqlite backend, artifacts in a mounted folder), `.env.example`, `.gitignore` that excludes `*.onnx`, `*.gguf`, `data/cache/`, `mlruns/`.
3. `tracking.py` with helpers to start a run and log the standard params from section 8 (seed, git SHA, model revision, package versions).
4. GitHub Actions `ci.yml` running `make lint` and `make test`.
5. README with a one-paragraph description and these headings only: Problem, Method, Results (generated), Reproduce, Limits.

**Constraints:** no model code yet.

**Accept:** `make lint && make test` pass; `docker compose up -d mlflow` serves the UI.

**Report:** file tree (depth 2) and command output.

---

## Phase 1

### Prompt 1.1: spike (decides the model and the export path)

**Read:** `CLAUDE.md` sections 3, 5, 7 step 5.

**Do:** write `scripts/spike_*.py` and `docs/SPIKE.md` answering, with evidence from running code (not from memory):
1. Does `Qwen/Qwen3-0.6B` export to ONNX through `onnxruntime-genai`'s model builder? Through Optimum? Record exact commands, package versions and errors. Repeat for `Qwen/Qwen3-1.7B` only if 0.6B works.
2. Does the exported model generate the same first 20 tokens as the PyTorch model on 5 prompts (greedy)?
3. How can different MatMul nodes get different precision in the installed ONNX Runtime version? Test: INT4 `MatMulNBits` on a chosen set of nodes with the rest left for INT8 dynamic quantization (`quantize_dynamic` with `nodes_to_quantize` or `nodes_to_exclude`). Show that two nodes can differ by loading the result and listing node op types.
4. Can the BFCL AST checker be imported from a pip package, or must the minimal matcher be written? Record what you found.
5. Decide the default model and export route; write the decision in `DECISIONS.md`. If no route works for Qwen3, test `Qwen/Qwen2.5-0.5B-Instruct` and `Qwen/Qwen2.5-1.5B-Instruct` and pick the first that works.

**Constraints:** do not build the project around an assumption the spike disproves; if a documented API differs from section 5 or 7, the spike result wins and `CLAUDE.md` gets a dated note.

**Accept:** `docs/SPIKE.md` has a table of attempts and a clear decision; a one-node-INT4, rest-INT8 ONNX model loads and runs.

**Report:** the decision in five lines and any change to `CLAUDE.md`.

### Prompt 1.2: scoring and data

**Read:** `CLAUDE.md` section 6.

**Do:**
1. `data/bfcl.py`: load the BFCL tasks needed (single-turn categories first) from a local checkout path set in `.env`; document in the README how to fetch the data.
2. `scoring/parse.py`: parse tool calls from the model's native format (verify the format from the tokenizer chat template for the chosen model).
3. `scoring/match.py`, `taxonomy.py`, `metrics.py`: implement the six metrics exactly as defined in section 6, using the BFCL checker if the spike allows. Unit tests with hand-written predictions covering each case (correct, wrong name, wrong argument, malformed JSON, no call when needed, call when not needed).
4. `data/splits.py` and committed `data/splits.json`: 100-prompt probe, rest validation, fixed seed, balanced categories. A test asserts no overlap.

**Constraints:** the scorer must be deterministic and fast (no model calls). No scoring definition may differ from section 6 without a `DECISIONS.md` entry.

**Accept:** `make test` passes with at least 25 scorer cases.

**Report:** the counts per category in probe and validation.

### Prompt 1.3: FP16 baseline run and statistics

**Read:** `CLAUDE.md` section 7 steps 1, 7, 8.

**Do:**
1. `evalrun/generate.py`: batched greedy generation with the chat template and tools; saves raw outputs per prompt.
2. `evalrun/run_eval.py`: run a configuration on probe or validation, compute metrics, write `results/eval_<config>_<split>_<utc>.json` and an MLflow run.
3. `evalrun/stats.py`: paired bootstrap (10,000 resamples, seeded) for a metric difference and exact McNemar, with tests against known cases (identical inputs give difference 0 with an interval containing 0).
4. Provide a notebook-friendly script `scripts/run_on_gpu.py` that runs the evaluation on a GPU machine and writes only result files.

**Constraints:** record the exact model revision hash. Generation must be reproducible from the seed.

**Accept:** you run the FP16 baseline (on GPU or CPU) and `results/` has the file; unit tests for the stats pass.

**Report:** FP16 baseline metrics with 95% intervals on probe and validation.

### Prompt 1.4: fake quantization and the probe

**Read:** `CLAUDE.md` sections 5 and 7 steps 2.

**Do:**
1. `quant/groups.py`: derive groups from the model structure (per decoder block; `attn_mlp` option). A test checks every linear weight in the decoder belongs to exactly one group and the embedding and `lm_head` belong to none.
2. `quant/fakequant.py`: per-group symmetric RTN for 4 and 8 bits. Tests: 8-bit error is smaller than 4-bit error on random weights; dequantized values lie on the grid; shape and dtype preserved; a group of identical values stays exact.
3. `quant/sizing.py`: the size model from section 5 with tests (all-INT4 smaller than all-INT8, budgets between them).
4. `quant/probe.py`: probe every group on the probe set, plus the perplexity probe on the fixed WikiText-2 slice. Save `results/sensitivity_<model>_<utc>.json` with both tables and a CSV.
5. A script to run the probe on a free GPU notebook, resumable per group (writes after each group).

**Constraints:** restore the original weights after each group (test that the model's output returns to baseline). Never quantize embeddings or `lm_head`.

**Accept:** unit tests pass; you run the probe and `results/` contains the sensitivity file with every group.

**Report:** the sensitivity table (group, tool-call drop, perplexity change) sorted by tool-call drop, and the rank correlation (Spearman) between the two columns.

### Prompt 1.5: allocator and maps

**Read:** `CLAUDE.md` section 7 step 3.

**Do:** implement `quant/allocate.py` producing `toolcall` and `ppl` maps at budgets `f = 0.25` and `f = 0.5`; write `maps/*.json` as specified. Tests: the size of every map is at or under its budget, a map assigns 4-bit to the lowest-sensitivity groups, the `toolcall` and `ppl` maps at the same budget have equal size within one group. Add a validation script that reports how many groups differ between the two maps.

**Constraints:** if the two maps are identical at a budget, say so and add budget points until they differ; do not force a difference.

**Accept:** maps committed and tests pass.

**Report:** for each budget, the number of groups at 4-bit and the overlap between the two maps.

### Prompt 1.6: export maps to ONNX

**Read:** `CLAUDE.md` section 7 step 5; `docs/SPIKE.md`.

**Do:**
1. `export/onnx_export.py` builds the FP ONNX model once (cached on disk).
2. `export/apply_map.py` applies a map: INT4 `MatMulNBits` for 4-bit groups and INT8 for the rest, using the route proven in the spike. It must also produce uniform INT8 and uniform INT4 models through the same code path.
3. Check that the node-to-group mapping matches the PyTorch groups; write `results/export_<map>_<utc>.json` with per-group op types and the file size. Compare the file size with the size model's prediction and record the gap.

**Constraints:** the build must be deterministic and scripted; no manual steps.

**Accept:** five ONNX models exist locally (INT8, INT4, `ppl`@0.25, `toolcall`@0.25, and the 0.5 pair as time allows); each loads and generates.

**Report:** a table of model, file size MB, predicted size MB, gap.

### Prompt 1.7: simulation versus real

**Read:** `CLAUDE.md` section 7 step 6.

**Do:** `export/parity.py`: for 200 prompts, compare greedy next-token agreement between the fake-quantized PyTorch model and the ONNX model for the same map; record agreement rate and first-divergence position per prompt. Also compare tool-call success on 100 validation prompts between the two for each map.

**Constraints:** if agreement is below 90%, investigate the cause (scale handling, group size, symmetric versus asymmetric) before proceeding, and record findings in `docs/notes.md`.

**Accept:** results file for each map.

**Report:** agreement table and the cause of any gap.

### Prompt 1.8: full evaluation and benchmark

**Read:** `CLAUDE.md` section 7 steps 7 to 9.

**Do:**
1. Run every configuration on the validation split with the ONNX models (CPU): FP16 reference (PyTorch), uniform INT8, uniform INT4, `ppl` and `toolcall` maps.
2. For every pair of interest compute differences with paired bootstrap intervals and McNemar p-values: `toolcall` versus `ppl` at the same budget; each map versus uniform INT4 and INT8.
3. `bench/speed.py` and `memory.py`: decode tokens per second, TTFT, p95 latency, peak resident memory; record the CPU model and thread count. Do not average across machines.
4. `make report`: README Results table and a size-versus-success chart (matplotlib, saved to `docs/`), generated from `results/` only.

**Constraints:** every number in the README comes from `make report`.

**Accept:** README has the results table; each configuration has an MLflow run.

**Report:** the full results table and one sentence stating whether H1 held, with the interval.

---

## Phase 2

### Prompt 2.1: interactions and swap search

**Read:** `CLAUDE.md` section 7 step 4.

**Do:** implement the interaction test (joint drop versus sum of single drops for the k least sensitive groups) and the capped swap local search on the probe set; confirm on validation only. Add more budget points (`f` in 0.1, 0.25, 0.5, 0.75).

**Constraints:** selection uses the probe set only; the validation score is reported, never optimised.

**Accept:** results for the interaction test and, if swaps improved the probe score, a refined map evaluated on validation.

**Report:** whether damage was additive, and the effect of the swap search with intervals.

### Prompt 2.2: multi-turn tasks and the llama.cpp baseline

**Do:**
1. Add the multi-turn BFCL categories to the scorer (per-call, per-trajectory and silent-failure rates, as separate fields) with tests.
2. Build the llama.cpp `Q4_K_M` baseline for the same model (document the exact conversion commands in `docs/llamacpp.md`), run the same prompts through its server or Python binding, and score with the same scorer.
3. Add both to the comparison table with intervals.

**Constraints:** llama.cpp results appear as a reference row at their own size, not as a same-size comparison.

**Accept:** at least 10 configurations in the report.

**Report:** the updated table.

### Prompt 2.3: serving, OpenTelemetry, Docker

**Read:** `CLAUDE.md` section 9.

**Do:** FastAPI server with ONNX Runtime and the best map; OpenTelemetry spans with the listed attributes; `docker/Dockerfile` (multi-stage, model not baked in, pulled from a path or bucket by environment variable); `/healthz` running one tool-call prompt; ONNX Runtime session settings from environment variables.

**Accept:** `docker build` and `docker run` serve a request returning a valid tool call; spans appear in a local Jaeger started from Compose.

**Report:** image size and one sample request and response.

### Prompt 2.4: ONNX Runtime tuning

**Do:** measure decode tokens per second, TTFT and peak memory while varying intra-op threads, graph optimisation level and IO binding for the KV cache. Choose the best settings, record before and after in `results/tuning_<utc>.json`, and make them the defaults.

**Constraints:** measure on the same machine, same prompt and output lengths as the Phase 1 benchmark.

**Accept:** the results file exists with the percentage change from the Phase 1 defaults.

**Report:** the table of settings and the change in tokens per second.

### Prompt 2.5: CI smoke eval

**Do:** a tiny configuration (50 probe prompts, one cached small ONNX model fetched in CI) that runs the scorer, a map application and a generation check in under 5 minutes. Add it to `ci.yml` and to `make check`. Cache models between runs.

**Accept:** CI is green; a deliberately broken export (wrong node mapping) fails the smoke eval.

**Report:** CI run times and the failing run URL.

### Prompt 2.6: a second model

**Do:** run the pipeline on a second small model that supports tool calling and exports through the same route. Produce maps, ONNX models, evaluation with intervals. Compare whether the sensitive groups are in similar positions and whether H1 holds on both.

**Constraints:** same code path, different config. Do not special-case the model.

**Accept:** the report has both models.

**Report:** a table with both models' key results and one paragraph on differences.

### Prompt 2.7: deploy plan, model card, write-up

**Do:**
1. Terraform in `infra/gcp` for Cloud Run (CPU, zero minimum instances) with the serving image from Artifact Registry and the model from a bucket. Do not apply; run `validate` and `plan` and stop.
2. Hugging Face model card text for the best map with results, intervals, the hardware, and limits.
3. `docs/WRITEUP.md`: question, method, results (generated), limits, prior art (the NF4 tool-call audit, the multi-turn BFCL study, TAQ, LieQ) with the differentiator in one sentence; state H1's outcome plainly.
4. `scripts/verify_claims.py` that extracts every number from README and WRITEUP and confirms each exists in `results/`.

**Constraints:** no "first" or "novel". Mention BFCL may be in the model's training data, so only same-model comparisons are meaningful.

**Accept:** `verify_claims.py` and `make check` pass.

**Report:** the final results summary and open questions.

---

## Utility prompts

### U1: review for correctness (after each phase)

Review `src/toolquant` against `CLAUDE.md` sections 5 to 7. List every place where probe and validation data could mix, where weights are not restored after a probe, where a number is computed with a different definition from section 6, where a result could be selected on validation, and where hardware is not recorded next to a speed number. For each finding give file and line, a failing test, and a fix. Do not change code until I approve the list.

### U2: debug a simulation-versus-real gap

Agreement between the fake-quantized PyTorch model and the ONNX model is `{paste}`. Isolate the cause by testing: one layer only, scale computation, group size, symmetric versus asymmetric, weight packing order, and activation dtype. Report which test explains the gap, add a regression test, and fix.

### U3: resume bullets from results only

Read `results/` and `README.md`. Write three resume bullets for ToolQuant, each under 30 words, using only numbers found in `results/` (file name in brackets after each for me to check, then I will remove the brackets). Use these exact terms where true: PyTorch, ONNX Runtime, quantization (PTQ), INT4, INT8, MLflow, OpenTelemetry, Docker, CI/CD, BFCL. State the CPU model next to every speed number. If H1 did not hold, write the bullet about the method and the measured finding instead. Do not use "first", "novel" or "cutting-edge".

### U4: interview preparation

Write 15 interview questions a senior ML engineer would ask about this project (why tool-call success as the objective, non-additive layer damage, group size and scales, fake versus real quantization, bootstrap intervals, contamination, what you would do next) with a 4-line model answer each, grounded in this repo's code and results.
