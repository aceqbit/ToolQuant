"""Run ToolQuant GPU steps on Kaggle/Colab and write only result files.

Notebook usage (one cell each):
    !git clone https://github.com/aceqbit/ToolQuant /kaggle/working/ToolQuant   # or /content/ToolQuant
    !python /kaggle/working/ToolQuant/scripts/run_on_gpu.py --task eval
    !python /kaggle/working/ToolQuant/scripts/run_on_gpu.py --task probe
Then download `toolquant_results.zip` (printed at the end) and unzip it into the repo's `results/`.

Tasks:
  eval   FP16 baseline on probe + validation   -> results/eval_fp16_*.json
  probe  per-group sensitivity (resumable)     -> results/sensitivity_*.json/.csv (+ partial file)
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "data" / "cache" / "qwen3-0.6b-hf"


def sh(*cmd: str) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def setup() -> None:
    os.environ["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")
    sys.path.insert(0, str(ROOT / "src"))
    sh(sys.executable, "-m", "pip", "install", "-q", "scipy", "statsmodels", "mlflow", "datasets")
    if not (ROOT / "data" / "cache" / "bfcl").exists():
        sh(sys.executable, "scripts/fetch_bfcl.py")
    if not MODEL_DIR.exists():
        from huggingface_hub import snapshot_download

        snapshot_download("Qwen/Qwen3-0.6B", local_dir=str(MODEL_DIR))


def bundle() -> None:
    out = shutil.make_archive(str(ROOT.parent / "toolquant_results"), "zip", ROOT, "results")
    print("DOWNLOAD THIS FILE AND UNZIP INTO results/:", out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["eval", "probe"], required=True)
    ap.add_argument("--batch-size", default="32")
    a = ap.parse_args()
    setup()
    py = sys.executable
    if a.task == "eval":
        for split in ("probe", "validation"):
            sh(py, "-m", "toolquant.evalrun.run_eval", "--config", "fp16", "--split", split,
               "--dtype", "float16", "--device", "cuda", "--batch-size", a.batch_size, "--no-mlflow")
    elif a.task == "probe":
        sh(py, "-m", "toolquant.quant.probe", "--device", "cuda", "--dtype", "float16",
           "--batch-size", a.batch_size)
    bundle()


if __name__ == "__main__":
    main()
