# SPIKE (Prompt 1.1)

Machine: Windows 11, Intel Core i5-1335U, 15.6 GB RAM, no GPU. Python 3.12 (uv).
Versions: onnxruntime 1.32.0, onnxruntime-genai 0.20.0, onnx 1.23.2, torch 2.15.0, transformers 5.24.1, numpy 2.5.5.
Model revision (HF cache metadata): `Qwen/Qwen3-0.6B` @ `c1899de289a04d12100db370d81485cdf75e47ca`.

## Attempts

| # | Question | Command / script | Result |
|---|---|---|---|
| 1 | Export Qwen3-0.6B via onnxruntime-genai builder | `python -m onnxruntime_genai.models.builder -m Qwen/Qwen3-0.6B -i data/cache/qwen3-0.6b-hf -o <dir> -p fp32 -e cpu` | **Works**, ~111 s, FP32 `model.onnx` + 2.4 GB external data, `genai_config.json`. |
| 2 | Export via Optimum | `optimum-cli export onnx --model ... --task text-generation-with-past <dir>` (optimum 2.3.0) | **Fails**: `optimum-cli: error: unrecognized arguments: onnx ...`. Optimum 2.x no longer ships the ONNX exporter (moved to a separate package). Not pursued; `optimum` removed from dependencies. |
| 3 | Qwen3-1.7B | not attempted | Skipped: 0.6B works, and 1.7B is slow on this CPU-only laptop. Spike decision is 0.6B. |
| 4 | Greedy parity, first 20 tokens, 5 prompts, PyTorch FP32 vs genai FP32 ONNX | `scripts/spike_gen.py data/cache/qwen3-0.6b-fp32` | Matching prefix 20/20, 20/20, 20/20, 3/3, 20/20 (the 4th prompt ends after 3 tokens in both). Mean matching prefix 16.6 (prompt 4 stops early). No divergence. |
| 5 | Per-node mixed precision | `scripts/spike_mixed.py`, inspected by `scripts/spike_inspect.py` | **Works** (see below). |
| 6 | BFCL AST checker as a pip package | `uv pip install --target <scratch> bfcl-eval` (2026.3.23) then import `bfcl_eval.eval_checker.ast_eval.ast_checker` | Package installs and ships the BFCL v4 data (`bfcl_eval/data/BFCL_v4_*.json`, `possible_answer/`), but importing the checker pulls in `qwen_agent` -> `soundfile` and a large LLM-client dependency tree (writer-sdk, tree-sitter, ...). Not importable in the project env without heavy extras. **Write the minimal matcher** (CLAUDE.md section 6). Use the pip package only as the *data source*. |

## Mixed precision route (item 3)

`quantize_dynamic` produces `MatMulInteger`, not `MatMulNBits`, and cannot do INT4. The route that works on ORT 1.32.0:

1. `MatMulNBitsQuantizer(model, bits=4, block_size=128, is_symmetric=True, nodes_to_exclude=<all nodes not in 4-bit set> + [lm_head])` writes INT4 for the chosen nodes only.
2. Load that result and run `MatMulNBitsQuantizer(model, bits=8, block_size=128, is_symmetric=True, nodes_to_exclude=[lm_head])`. Already-quantized `MatMulNBits` nodes are not MatMul, so only the remaining MatMul nodes become 8-bit `MatMulNBits`.
3. `lm_head` is left as FP MatMul in both passes (CLAUDE.md: not searched).

Evidence (`scripts/spike_inspect.py data/cache/spike_mixed`, layer 0 at 4-bit, others 8-bit):

```
('layers.0', 'MatMulNBits', 4) 4
('other',    'MatMul', 0)       1      # lm_head
('other',    'MatMulNBits', 8)  108
```

The model loads in onnxruntime-genai and generates coherent text ("The capital of France is" -> " Paris. ...").
Note: the genai builder emits 4 MatMul nodes per decoder layer (28 x 4 = 112), not 7; node-to-group mapping must use the exported node names (`/model/layers.N/...`).
Size: stage-4 intermediate 2.35 GB; final mixed model 1.09 GB (data) vs 2.41 GB FP32.

## Decision

- Model: `Qwen/Qwen3-0.6B`.
- Export: onnxruntime-genai model builder (FP32, CPU) once, then ORT `MatMulNBitsQuantizer` two-pass (4-bit on the chosen nodes, 8-bit on the rest) via `nodes_to_exclude`.
- INT8 uses 8-bit `MatMulNBits` (block 128), not `quantize_dynamic`; this matches the fake quantizer (weight-only, per-group symmetric).
- Scorer: minimal matcher; BFCL v4 data taken from the `bfcl-eval` wheel's data files.
