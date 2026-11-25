"""Check that every number in README.md and docs/WRITEUP.md exists in results/.

Numbers inside code fences, inline code, URLs, file names and headings are ignored. Spec constants
(split sizes, bootstrap resamples, ...) are listed in docs/claims_allow.txt (one number per line).
A result number matches if the document shows it at the precision the document used, either as
the raw value, x100 (percent), or as a rounded difference.

Exit status 1 and a list of unmatched numbers if any are missing.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

NUM = re.compile(r"(?<![\w./-])[-+]?\d[\d,]*\.?\d*")


def collect(obj, out: set[float]) -> None:
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        out.add(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            collect(v, out)
    elif isinstance(obj, list):
        for v in obj:
            collect(v, out)


def result_numbers(results: Path = Path("results")) -> set[float]:
    nums: set[float] = set()
    for p in results.glob("*.json"):
        if p.name.startswith("sensitivity_partial"):
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        d.pop("records", None) if isinstance(d, dict) else None
        collect(d, nums)
        # per-prompt records: add aggregate counts only (already in summary)
    for p in results.glob("eval_*.json"):
        d = json.loads(p.read_text(encoding="utf-8"))
        collect(d.get("summary", {}), nums)
    return nums


def allowed(path: Path = Path("docs/claims_allow.txt")) -> set[float]:
    if not path.exists():
        return set()
    return {float(x) for x in path.read_text().split() if x.strip()}


def strip_ignored(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"`[^`]*`", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\]\([^)]*\)", "]", text)
    text = re.sub(r"^#.*$", " ", text, flags=re.M)
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.S)
    text = re.sub(r"\b[\w.-]+\.(?:json|md|py|csv|png|onnx)\b", " ", text)
    text = re.sub(
        r"\b(?:Qwen\d?[\w.-]*|Q\d_K_M|INT\d|FP\d+|BFCL[\w.]*|v\d+|p\d-\d|H\d)\b", " ", text
    )
    text = re.sub(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{8}T\d{6}Z\b", " ", text)
    return text


def matches(token: float, decimals: int, pool: set[float]) -> bool:
    for r in pool:
        for cand in (r, r * 100):
            if round(cand, decimals) == round(token, decimals):
                return True
            if abs(round(cand, decimals) - abs(token)) < 1e-9:
                return True
    return False


def check(files: list[Path], results: Path = Path("results")) -> list[tuple[str, str]]:
    pool = result_numbers(results)
    allow = allowed()
    bad = []
    for f in files:
        if not f.exists():
            continue
        for tok in NUM.findall(strip_ignored(f.read_text(encoding="utf-8"))):
            clean = tok.replace(",", "").rstrip(".")
            if not clean or clean in "+-":
                continue
            val = float(clean)
            dec = len(clean.split(".")[1]) if "." in clean else 0
            if val in allow or abs(val) in allow:
                continue
            if not matches(val, dec, pool):
                bad.append((f.name, tok))
    return bad


if __name__ == "__main__":
    files = [Path("README.md"), Path("docs/WRITEUP.md")]
    missing = check(files)
    for f, t in missing:
        print(f"{f}: number {t} not found in results/")
    print("OK" if not missing else f"{len(missing)} unmatched numbers")
    sys.exit(1 if missing else 0)
