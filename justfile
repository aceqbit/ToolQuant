set shell := ["powershell.exe", "-NoProfile", "-Command"]

default:
    just --list

# Install Python 3.12 environment
env:
    uv sync

# Start the local MLflow server (native, no Docker)
mlflow:
    uv run mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts

# CPU probe (slow: see docs/GPU.md for the GPU route)
probe:
    uv run python -m toolquant.quant.probe

allocate SENS:
    uv run python -m toolquant.quant.allocate --sensitivity {{SENS}}

# MAP is a maps/*.json path, or int4 / int8
export MAP:
    uv run python -m toolquant.export.apply_map --map {{MAP}}

parity MAP:
    uv run python -m toolquant.export.parity --map {{MAP}}

# NAMES are directories under data/cache/onnx (validation split, CPU)
eval *NAMES:
    uv run python scripts/run_onnx_evals.py {{NAMES}}

bench NAME:
    uv run python -m toolquant.bench.speed --onnx-dir data/cache/onnx/{{NAME}} --name {{NAME}}

serve:
    @Write-Error "not implemented: serve"; exit 1

report:
    uv run python -m toolquant.report

lint:
    uv run ruff check src tests
    uv run ruff format --check src tests

test:
    uv run pytest -q

# lint, tests; smoke eval is added in prompt 2.5
check: lint test
