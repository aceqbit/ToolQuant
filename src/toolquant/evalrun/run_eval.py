"""Run one configuration on one split; write results/eval_<config>_<split>_<utc>.json + MLflow."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from toolquant.data.bfcl import CATEGORIES, Task, load_all
from toolquant.data.splits import load_splits, select
from toolquant.evalrun.stats import bootstrap_ci
from toolquant.scoring.metrics import aggregate, score_output
from toolquant.tracking import git_sha, package_versions

MODEL_ID = "Qwen/Qwen3-0.6B"
DEFAULT_HF_PATH = "data/cache/qwen3-0.6b-hf"
RESULTS = Path("results")
PER_PROMPT_METRICS = ("tool_call_success", "missed_call", "silent_failure")


def model_revision(path: str) -> str:
    """Commit hash recorded by huggingface_hub next to the downloaded files."""
    meta = Path(path) / ".cache" / "huggingface" / "download" / "config.json.metadata"
    if meta.exists():
        return meta.read_text().splitlines()[0].strip()
    return os.environ.get("MODEL_REVISION", "unknown")


def hardware() -> dict[str, Any]:
    cpu = platform.processor()
    if platform.system() == "Windows":
        try:
            import subprocess

            cpu = (
                subprocess.run(
                    [
                        "powershell",
                        "-NoProfile",
                        "-Command",
                        "(Get-CimInstance Win32_Processor).Name",
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
                or cpu
            )
        except OSError:
            pass
    elif Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    gpu = None
    try:
        import torch

        threads = torch.get_num_threads()
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
    except ImportError:
        threads = os.cpu_count()
    return {"cpu": cpu, "logical_cpus": os.cpu_count(), "torch_threads": threads, "gpu": gpu}


def score_texts(tasks: list[Task], texts: dict[str, str]) -> list[dict[str, Any]]:
    recs = []
    for t in tasks:
        r = score_output(t, texts[t.id])
        r["raw_output"] = texts[t.id]
        recs.append(r)
    return recs


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    agg = aggregate(records)
    agg["intervals"] = {}
    for m in PER_PROMPT_METRICS:
        mean, lo, hi = bootstrap_ci([r["outcomes"][m] for r in records])
        agg["intervals"][m] = {"mean": mean, "lo": lo, "hi": hi}
    return agg


def write_result(
    config: str,
    split: str,
    records: list[dict[str, Any]],
    meta: dict[str, Any],
    results_dir: Path = RESULTS,
    log_mlflow: bool = True,
) -> Path:
    summary = summarize(records)
    utc = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "config": config,
        "split": split,
        "utc": utc,
        "git_sha": git_sha(),
        "packages": package_versions(),
        "hardware": hardware(),
        "summary": summary,
        **meta,
        "records": records,
    }
    results_dir.mkdir(exist_ok=True)
    path = results_dir / f"eval_{config}_{split}_{utc}.json"
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    if log_mlflow:
        log_to_mlflow(config, split, summary, payload, path)
    return path


def log_to_mlflow(
    config: str, split: str, summary: dict[str, Any], payload: dict[str, Any], path: Path
) -> None:
    import mlflow

    from toolquant.tracking import start_run

    with start_run(
        f"{config}-{split}",
        model=payload.get("model", MODEL_ID),
        revision=payload.get("revision", "unknown"),
        seed=payload.get("seed", 0),
        group_size=payload.get("group_size"),
        granularity=payload.get("granularity"),
        budget=payload.get("budget"),
        map_kind=payload.get("map_kind"),
    ):
        mlflow.log_params({"config": config, "split": split, "n": summary["n"]})
        for m, v in summary["intervals"].items():
            mlflow.log_metrics({m: v["mean"], f"{m}_lo": v["lo"], f"{m}_hi": v["hi"]})
        for k in ("ast_match", "abstain_ok", "schema_valid"):
            if summary[k] is not None:
                mlflow.log_metric(k, summary[k])
        for k, v in payload.get("extra_metrics", {}).items():
            mlflow.log_metric(k, v)
        mlflow.log_artifact(str(path))


def get_tasks(split: str, limit: int | None = None) -> list[Task]:
    tasks = load_all(CATEGORIES)
    sel = select(tasks, load_splits()[split])
    return sel[:limit] if limit else sel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="name, e.g. fp32")
    ap.add_argument("--split", choices=["probe", "validation"], required=True)
    ap.add_argument("--backend", choices=["hf", "onnx"], default="hf")
    ap.add_argument("--onnx-dir", help="model directory for --backend onnx")
    ap.add_argument("--map", help="map JSON (recorded as run params)")
    ap.add_argument("--model-path", default=DEFAULT_HF_PATH)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--no-mlflow", action="store_true")
    a = ap.parse_args()

    from toolquant.evalrun.generate import generate_hf, generate_onnx, load_hf

    tasks = get_tasks(a.split, a.limit)
    t0 = time.time()

    def prog(done: int, total: int) -> None:
        print(f"{done}/{total} {time.time() - t0:.0f}s", flush=True)

    if a.backend == "hf":
        model, tok = load_hf(a.model_path, a.dtype, a.device)
        texts = generate_hf(model, tok, tasks, a.batch_size, progress=prog)
        n_params = sum({id(p): p.numel() for p in model.parameters()}.values())
        size_fp16_mb = n_params * 2 / 1e6  # FP16-equivalent size of the unquantized model
    else:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(a.model_path)
        texts = generate_onnx(a.onnx_dir, tok, tasks, progress=prog)
    meta: dict[str, Any] = {
        "model": MODEL_ID,
        "revision": model_revision(a.model_path),
        "seed": 0,
        "dtype": a.dtype,
        "backend": a.backend,
        "gen_seconds": time.time() - t0,
    }
    if a.backend == "hf":
        meta["extra_metrics"] = {"size_mb": size_fp16_mb}
    if a.onnx_dir:
        d = Path(a.onnx_dir)
        meta["onnx_dir"] = str(d)
        meta["onnx_size_mb"] = sum(p.stat().st_size for p in d.iterdir() if "onnx" in p.name) / 1e6
        meta["extra_metrics"] = {"size_mb": meta["onnx_size_mb"]}
    if a.map:
        mp = json.loads(Path(a.map).read_text(encoding="utf-8"))
        meta.update(
            map_kind=mp.get("kind"),
            budget=mp.get("budget_f"),
            group_size=mp.get("group_size"),
            granularity=mp.get("granularity"),
            map_file=a.map,
        )
    path = write_result(
        a.config, a.split, score_texts(tasks, texts), meta, log_mlflow=not a.no_mlflow
    )
    print("wrote", path)


if __name__ == "__main__":
    main()
