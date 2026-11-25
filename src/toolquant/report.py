"""`just report`: README results tables and the size-vs-success chart, from results/ only."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from toolquant.evalrun.stats import compare

RESULTS = Path("results")
START, END = "<!-- results:start -->", "<!-- results:end -->"


def latest(pattern: str, results: Path = RESULTS) -> dict[str, Path]:
    """Latest file per config name for files like eval_<config>_<split>_<utc>.json."""
    best: dict[str, Path] = {}
    for p in sorted(results.glob(pattern)):
        m = re.match(
            r"(?:eval|bench|parity|export)_(.+?)(?:_(?:probe|validation))?_\d{8}T\d{6}Z", p.stem
        )
        if m:
            best[m.group(1)] = p
    return best


def load(p: Path) -> dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def config_order(name: str) -> tuple[int, float, str]:
    if name in ("fp16", "fp32"):
        return (0, 0, name)
    if name == "int8":
        return (1, 0, name)
    if name == "int4":
        return (2, 0, name)
    m = re.search(r"_([0-9.]+)_(toolcall|ppl)$", name)
    return (3, float(m.group(1)) if m else 9, name)


def fmt(m: dict[str, float], pct: bool = True) -> str:
    k = 100 if pct else 1
    return f"{m['mean'] * k:.1f} [{m['lo'] * k:.1f}, {m['hi'] * k:.1f}]"


def build_tables(results: Path = RESULTS) -> tuple[str, list[dict[str, Any]]]:
    evals = {k: load(v) for k, v in latest("eval_*_validation_*.json", results).items()}
    bench = {k: load(v) for k, v in latest("bench_*.json", results).items()}
    parity = {k: load(v) for k, v in latest("parity_*.json", results).items()}
    names = sorted(evals, key=config_order)
    rows = []
    lines = [
        (
            "| Config | Size MB | Success % [95% CI] | Missed call % [95% CI] | "
            "Silent failure % [95% CI] | Decode tok/s | TTFT ms | p95 latency s | Peak RSS MB | "
            "Parity (next-token) |"
        ),
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for n in names:
        e = evals[n]
        iv = e["summary"]["intervals"]
        size = e.get("extra_metrics", {}).get("size_mb")
        b = bench.get(n)
        bs = b["summary"] if b else None
        par = parity.get(n)
        lines.append(
            "| {n} | {size} | {s} | {m} | {sf} | {tps} | {ttft} | {p95} | {rss} | {par} |".format(
                n=n,
                size=f"{size:.0f}" if size else "-",
                s=fmt(iv["tool_call_success"]),
                m=fmt(iv["missed_call"]),
                sf=fmt(iv["silent_failure"]),
                tps=f"{bs['decode_tokens_per_s_mean']:.1f}" if bs else "-",
                ttft=f"{bs['ttft_ms_mean']:.0f}" if bs else "-",
                p95=f"{bs['latency_s_p95']:.2f}" if bs else "-",
                rss=f"{b['peak_rss_mb']:.0f}" if b else "-",
                par=f"{par['agreement']['next_token_agreement'] * 100:.1f}%" if par else "-",
            )
        )
        rows.append({"name": n, "size": size, "iv": iv["tool_call_success"]})
    hw = next((b["hardware"] for b in bench.values()), None)
    out = "\n".join(lines)
    if hw:
        out += (
            f"\n\nSpeed measured on: {hw['cpu']} ({hw['logical_cpus']} logical CPUs), "
            "ONNX Runtime CPU, 128-token prompt, 128 output tokens, 5 warm-up + 20 timed runs."
        )
    out += "\n\n" + pair_table(evals)
    return out, rows


def _outcomes(e: dict[str, Any], metric: str) -> dict[str, bool]:
    return {r["id"]: r["outcomes"][metric] for r in e["records"]}


def pair_table(evals: dict[str, dict[str, Any]]) -> str:
    pairs: list[tuple[str, str]] = []
    budgets = sorted(
        {m.group(1) for n in evals if (m := re.search(r"_([0-9.]+)_toolcall$", n))}, key=float
    )
    for f in budgets:
        tc = next(n for n in evals if n.endswith(f"_{f}_toolcall"))
        pp = next((n for n in evals if n.endswith(f"_{f}_ppl")), None)
        if pp:
            pairs.append((tc, pp))
        for ref in ("int4", "int8"):
            if ref in evals:
                pairs.append((tc, ref))
                if pp:
                    pairs.append((pp, ref))
    lines = [
        (
            "Differences in tool-call success (percentage points; paired bootstrap, 10,000 "
            "resamples, 95% interval; exact McNemar p):"
        ),
        "",
        "| A | B | A - B (pp) [95% CI] | McNemar p | A only correct | B only correct |",
        "|---|---|---|---|---|---|",
    ]
    for a, b in pairs:
        oa, ob = _outcomes(evals[a], "tool_call_success"), _outcomes(evals[b], "tool_call_success")
        ids = sorted(set(oa) & set(ob))
        r = compare([oa[i] for i in ids], [ob[i] for i in ids])
        lines.append(
            f"| {a} | {b} | {r['diff'] * 100:+.1f} [{r['lo'] * 100:+.1f}, {r['hi'] * 100:+.1f}] | "
            f"{r['p']:.3f} | {r['n10']} | {r['n01']} |"
        )
    return "\n".join(lines)


def chart(rows: list[dict[str, Any]], path: Path = Path("docs/size_vs_success.png")) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = [r for r in rows if r["size"]]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for r in pts:
        iv = r["iv"]
        kind = (
            "toolcall"
            if r["name"].endswith("toolcall")
            else "ppl"
            if r["name"].endswith("ppl")
            else "uniform"
        )
        color = {"toolcall": "#1f77b4", "ppl": "#d62728", "uniform": "#555555"}[kind]
        ax.errorbar(
            r["size"],
            iv["mean"] * 100,
            yerr=[[(iv["mean"] - iv["lo"]) * 100], [(iv["hi"] - iv["mean"]) * 100]],
            fmt="o",
            color=color,
            capsize=3,
        )
        ax.annotate(
            r["name"].replace("qwen3-0.6b_", ""),
            (r["size"], iv["mean"] * 100),
            fontsize=7,
            xytext=(4, 4),
            textcoords="offset points",
        )
    ax.set_xlabel("model size (MB)")
    ax.set_ylabel("tool-call success on validation (%)")
    ax.set_title("Size vs tool-call success (95% bootstrap CI)")
    ax.grid(alpha=0.3)
    path.parent.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    table, rows = build_tables()
    readme = Path("README.md")
    text = readme.read_text(encoding="utf-8")
    block = f"{START}\n\n{table}\n\n![size vs success](docs/size_vs_success.png)\n\n{END}"
    if START in text:
        text = re.sub(
            re.escape(START) + r".*?" + re.escape(END), lambda _: block, text, flags=re.DOTALL
        )
    else:
        text = text.replace("## Results (generated)", f"## Results (generated)\n\n{block}", 1)
    readme.write_text(text, encoding="utf-8")
    chart(rows)
    print("README results updated from", len(rows), "configurations")


if __name__ == "__main__":
    main()
