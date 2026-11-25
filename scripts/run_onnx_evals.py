"""Run ONNX evals (validation split) one after another; skip configs that already have a result.

    uv run python scripts/run_onnx_evals.py int8 int4 qwen3-0.6b_0.25_toolcall ...

Each name is a directory under data/cache/onnx/. Map-based names also pass the map JSON from maps/.
"""

import subprocess
import sys
from pathlib import Path


def done(name: str, split: str) -> bool:
    return any(Path("results").glob(f"eval_{name}_{split}_*.json"))


def main(names: list[str], split: str = "validation") -> None:
    for name in names:
        if done(name, split):
            print("skip", name, flush=True)
            continue
        cmd = [
            sys.executable, "-m", "toolquant.evalrun.run_eval",
            "--config", name, "--split", split,
            "--backend", "onnx", "--onnx-dir", f"data/cache/onnx/{name}",
        ]  # fmt: skip
        mp = Path("maps") / f"{name}.json"
        if mp.exists():
            cmd += ["--map", str(mp)]
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main(sys.argv[1:])
