set shell := ["powershell.exe", "-NoProfile", "-Command"]

default:
    just --list

# Install Python 3.12 environment
env:
    uv sync

# Start the local MLflow server (native, no Docker)
mlflow:
    uv run mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts

probe:
    @Write-Error "not implemented: probe"; exit 1

allocate:
    @Write-Error "not implemented: allocate"; exit 1

export MAP="":
    @Write-Error "not implemented: export"; exit 1

parity:
    @Write-Error "not implemented: parity"; exit 1

eval:
    @Write-Error "not implemented: eval"; exit 1

bench:
    @Write-Error "not implemented: bench"; exit 1

serve:
    @Write-Error "not implemented: serve"; exit 1

report:
    @Write-Error "not implemented: report"; exit 1

lint:
    uv run ruff check src tests
    uv run ruff format --check src tests

test:
    uv run pytest -q

# lint, tests; smoke eval is added in prompt 2.5
check: lint test
