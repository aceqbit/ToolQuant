# DECISIONS

## 2026-11-25 Environment substitutions (Prompt 0)
- Native Windows 11 + PowerShell: `just` replaces `make` (CLAUDE.md "make X" == `just X`). `just check` = lint + test.
- No Docker: MLflow runs natively (`just mlflow`, sqlite backend, `./mlartifacts`); no docker-compose.yml, Dockerfile or Terraform; llama.cpp baseline skipped.
- Dependency `mlflow` added now (section 3 lists MLflow; needed by tracking.py). Dev: pytest, ruff, mypy.
- CI uses `just` via extractions/setup-just.
