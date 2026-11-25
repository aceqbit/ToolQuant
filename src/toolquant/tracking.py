"""MLflow helpers: start a run and log the standard params (CLAUDE.md section 8)."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from importlib import metadata
from typing import Any

EXPERIMENT = "toolquant"
TRACKED_PACKAGES = (
    "torch",
    "transformers",
    "datasets",
    "onnxruntime",
    "onnxruntime-genai",
    "optimum",
    "numpy",
    "scipy",
    "statsmodels",
    "mlflow",
)


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def package_versions(names: tuple[str, ...] = TRACKED_PACKAGES) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return versions


def standard_params(
    *,
    model: str,
    revision: str,
    seed: int,
    group_size: int | None = None,
    granularity: str | None = None,
    budget: float | None = None,
    map_kind: str | None = None,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "model": model,
        "revision": revision,
        "seed": seed,
        "git_sha": git_sha(),
    }
    optional = {
        "group_size": group_size,
        "granularity": granularity,
        "budget": budget,
        "map_kind": map_kind,
    }
    params.update({k: v for k, v in optional.items() if v is not None})
    for pkg, ver in package_versions().items():
        params[f"pkg.{pkg}"] = ver
    return params


@contextmanager
def start_run(run_name: str, **param_kwargs: Any) -> Iterator[Any]:
    """Start an MLflow run in experiment `toolquant` with the standard params logged."""
    import mlflow

    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if uri:
        mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(standard_params(**param_kwargs))
        yield run
