"""Build the FP ONNX model once with the onnxruntime-genai builder (cached on disk)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

DEFAULT_HF = Path("data/cache/qwen3-0.6b-hf")
DEFAULT_FP = Path("data/cache/onnx/fp32")


def build_fp_onnx(
    hf_dir: Path = DEFAULT_HF, out_dir: Path = DEFAULT_FP, model_id: str = "Qwen/Qwen3-0.6B"
) -> Path:
    """FP32 CPU export (the route proven in docs/SPIKE.md). Returns the output directory."""
    if (out_dir / "model.onnx").exists() and (out_dir / "genai_config.json").exists():
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "onnxruntime_genai.models.builder",
            "-m",
            model_id,
            "-i",
            str(hf_dir),
            "-o",
            str(out_dir),
            "-p",
            "fp32",
            "-e",
            "cpu",
        ],
        check=True,
    )
    return out_dir


if __name__ == "__main__":
    print(build_fp_onnx())
