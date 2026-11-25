"""Apply a precision map to the FP ONNX model (route proven in docs/SPIKE.md).

Pass 1: 4-bit `MatMulNBitsQuantizer` on the 4-bit groups' nodes (all other nodes excluded).
Pass 2: 8-bit `MatMulNBitsQuantizer` on every remaining MatMul except `lm_head`.
Uniform INT4 and INT8 go through the same function with an all-4 / all-8 map.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import onnx
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

from toolquant.export.onnx_export import DEFAULT_FP, build_fp_onnx

_NODE = re.compile(r"/layers\.(\d+)/(attn|mlp)/")
ONNX_ROOT = Path("data/cache/onnx")
RESULTS = Path("results")


def node_groups(model: onnx.ModelProto, granularity: str = "block") -> dict[str, list[str]]:
    """Map PyTorch-style group name -> ONNX MatMul node names (lm_head excluded)."""
    groups: dict[str, list[str]] = {}
    for n in model.graph.node:
        if n.op_type not in ("MatMul", "MatMulNBits"):
            continue
        m = _NODE.search(n.name)
        if not m:
            continue
        idx, part = int(m.group(1)), m.group(2)
        key = f"layer{idx:02d}" if granularity == "block" else f"layer{idx:02d}.{part}"
        groups.setdefault(key, []).append(n.name)
    return dict(sorted(groups.items()))


def group_weight_params(model: onnx.ModelProto, groups: dict[str, list[str]]) -> dict[str, int]:
    """Weight element counts per group, from FP initializer shapes."""
    init = {i.name: i for i in model.graph.initializer}
    by_name = {n.name: n for n in model.graph.node}
    out = {}
    for g, names in groups.items():
        total = 0
        for nm in names:
            w = init[by_name[nm].input[1]]
            n = 1
            for d in w.dims:
                n *= d
            total += n
        out[g] = total
    return out


def _quant(src: Path, dst: Path, bits: int, exclude: list[str], block_size: int) -> None:
    model = onnx.load(str(src / "model.onnx"))
    q = MatMulNBitsQuantizer(
        model, bits=bits, block_size=block_size, is_symmetric=True, nodes_to_exclude=exclude
    )
    q.process()
    dst.mkdir(parents=True, exist_ok=True)
    q.model.save_model_to_file(str(dst / "model.onnx"), use_external_data_format=True)


def _copy_support(src: Path, dst: Path) -> None:
    for p in src.iterdir():
        if p.suffix in {".json", ".jinja"}:
            shutil.copy(p, dst / p.name)


def apply_map(
    bits_per_group: dict[str, int],
    out_dir: Path,
    fp_dir: Path = DEFAULT_FP,
    granularity: str = "block",
    block_size: int = 128,
) -> Path:
    """Build the mixed-precision ONNX model for `bits_per_group` into out_dir."""
    if (out_dir / "model.onnx").exists():
        shutil.rmtree(out_dir)
    fp = onnx.load(str(fp_dir / "model.onnx"), load_external_data=False)
    groups = node_groups(fp, granularity)
    missing = set(groups) ^ set(bits_per_group)
    if missing:
        raise ValueError(f"map and ONNX groups differ: {sorted(missing)}")
    all_nodes = [n for names in groups.values() for n in names]
    four = [n for g, names in groups.items() if bits_per_group[g] == 4 for n in names]
    eight = [n for g, names in groups.items() if bits_per_group[g] == 8 for n in names]
    if set(bits_per_group.values()) - {4, 8}:
        raise ValueError("bits must be 4 or 8")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out_dir.parent) as tmp:
        cur = fp_dir
        if four:
            stage = Path(tmp) / "s4"
            _quant(
                cur,
                stage,
                4,
                [n for n in all_nodes if n not in four] + ["/lm_head/MatMul"],
                block_size,
            )
            cur = stage
        if eight:
            stage = Path(tmp) / "s8"
            _quant(cur, stage, 8, ["/lm_head/MatMul"], block_size)
            cur = stage
        out_dir.mkdir(parents=True, exist_ok=True)
        for p in cur.iterdir():
            if p.suffix == ".onnx" or p.name.endswith(".onnx.data"):
                shutil.move(str(p), out_dir / p.name)
    _copy_support(fp_dir, out_dir)
    return out_dir


def inspect(model_dir: Path, granularity: str = "block") -> dict[str, Any]:
    """Per-group op types and bits of a built model, plus total file size in bytes."""
    m = onnx.load(str(model_dir / "model.onnx"), load_external_data=False)
    groups = node_groups(m, granularity)
    by_name = {n.name: n for n in m.graph.node}
    per_group: dict[str, Any] = {}
    for g, names in groups.items():
        ops = sorted(
            {
                (
                    by_name[n].op_type,
                    next((a.i for a in by_name[n].attribute if a.name == "bits"), 0),
                )
                for n in names
            }
        )
        per_group[g] = [{"op": o, "bits": b} for o, b in ops]
    size = sum(p.stat().st_size for p in model_dir.iterdir() if "onnx" in p.name)
    return {"per_group": per_group, "file_bytes": size}


def build_named(
    name: str, bits_per_group: dict[str, int], granularity: str = "block", **kw: Any
) -> Path:
    fp_dir = build_fp_onnx()
    return apply_map(bits_per_group, ONNX_ROOT / name, fp_dir, granularity, **kw)


def write_report(
    name: str,
    bits_per_group: dict[str, int],
    model_dir: Path,
    pt_group_params: dict[str, int],
    predicted_bits: float,
    pt_decoder_predicted_bits: float,
    granularity: str = "block",
    fp_dir: Path = DEFAULT_FP,
) -> Path:
    fp = onnx.load(str(fp_dir / "model.onnx"), load_external_data=False)
    onnx_params = group_weight_params(fp, node_groups(fp, granularity))
    info = inspect(model_dir, granularity)
    wrong = [
        g
        for g, ops in info["per_group"].items()
        if any(o["op"] != "MatMulNBits" or o["bits"] != bits_per_group[g] for o in ops)
    ]
    actual_mb = info["file_bytes"] / 1e6
    pred_mb = predicted_bits / 8 / 1e6
    utc = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    rep = {
        "name": name,
        "utc": utc,
        "bits_per_group": bits_per_group,
        "per_group": info["per_group"],
        "groups_with_wrong_precision": wrong,
        "group_param_match": {
            g: {"pytorch": pt_group_params.get(g), "onnx": onnx_params.get(g)} for g in onnx_params
        },
        "param_mismatch_groups": [
            g for g in onnx_params if onnx_params[g] != pt_group_params.get(g)
        ],
        "file_mb": actual_mb,
        "predicted_mb_fp16_other": pred_mb,
        "gap_mb": actual_mb - pred_mb,
        "gap_ratio": actual_mb / pred_mb,
        "predicted_decoder_mb": pt_decoder_predicted_bits / 8 / 1e6,
        "note": "ONNX build keeps embedding and lm_head in FP32 and scales in FP32.",
    }
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"export_{name}_{utc}.json"
    path.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    return path


def main() -> None:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    from toolquant.quant.groups import group_info
    from toolquant.quant.sizing import total_bits

    ap = argparse.ArgumentParser()
    ap.add_argument("--map", help="map JSON path, or 'int4' / 'int8'", required=True)
    ap.add_argument("--granularity", default="block")
    a = ap.parse_args()
    cfg = AutoConfig.from_pretrained("data/cache/qwen3-0.6b-hf")
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg)
    info = group_info(model, a.granularity)
    if a.map in ("int4", "int8"):
        name, bits = a.map, {g: int(a.map[3:]) for g in info["params"]}
    else:
        mp = json.loads(Path(a.map).read_text(encoding="utf-8"))
        name, bits = Path(a.map).stem, mp["bits_per_group"]
    out = build_named(name, bits, a.granularity)
    pred = total_bits(info["params"], bits, info["other_params"])
    pred_dec = total_bits(info["params"], bits, 0)
    print(write_report(name, bits, out, info["params"], pred, pred_dec, a.granularity))


if __name__ == "__main__":
    main()
