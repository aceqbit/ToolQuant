"""Spike: per-node mixed precision with MatMulNBitsQuantizer (4-bit pass, then 8-bit pass)."""

import sys
import time
from pathlib import Path

import onnx
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

SRC = Path("data/cache/qwen3-0.6b-fp32/model.onnx")
OUT = Path("data/cache/spike_mixed")


def matmul_names(model_path: Path) -> list[str]:
    m = onnx.load(str(model_path), load_external_data=False)
    return [n.name for n in m.graph.node if n.op_type == "MatMul" and "lm_head" not in n.name]


def quant(src: Path, dst: Path, bits: int, exclude: list[str]) -> None:
    m = onnx.load(str(src))
    q = MatMulNBitsQuantizer(
        m, bits=bits, block_size=128, is_symmetric=True, nodes_to_exclude=exclude
    )
    q.process()
    dst.parent.mkdir(parents=True, exist_ok=True)
    q.model.save_model_to_file(str(dst), use_external_data_format=True)


if __name__ == "__main__":
    names = matmul_names(SRC)
    four = [n for n in names if "/layers.0/" in n]
    eight = [n for n in names if n not in four]
    t = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    quant(SRC, OUT / "stage4" / "model.onnx", 4, eight + ["/lm_head/MatMul"])
    print("4-bit pass", time.time() - t, file=sys.stderr)
    quant(OUT / "stage4" / "model.onnx", OUT / "model.onnx", 8, ["/lm_head/MatMul"])
    print("8-bit pass", time.time() - t, file=sys.stderr)
    m = onnx.load(str(OUT / "model.onnx"), load_external_data=False)
    for n in m.graph.node:
        if n.op_type == "MatMulNBits" and ("layers.0/" in n.name or "layers.1/" in n.name):
            bits = next(a.i for a in n.attribute if a.name == "bits")
            print(n.name, n.op_type, "bits=", bits)
        if n.op_type == "MatMul":
            print("left as MatMul:", n.name)
