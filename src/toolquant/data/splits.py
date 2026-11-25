"""Fixed-seed probe/validation split. The split file `data/splits.json` is committed."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from toolquant.data.bfcl import CATEGORIES, Task, load_all

SPLITS_PATH = Path("data/splits.json")
SEED = 0
PROBE_SIZE = 100


def make_splits(
    tasks: list[Task], seed: int = SEED, probe_size: int = PROBE_SIZE
) -> dict[str, Any]:
    cats = list(dict.fromkeys(t.category for t in tasks))
    base, extra = divmod(probe_size, len(cats))
    probe: list[str] = []
    validation: list[str] = []
    rng = random.Random(seed)
    for i, c in enumerate(cats):
        ids = sorted(t.id for t in tasks if t.category == c)
        rng.shuffle(ids)
        k = base + (1 if i < extra else 0)
        probe += ids[:k]
        validation += ids[k:]
    return {
        "seed": seed,
        "categories": cats,
        "probe": sorted(probe),
        "validation": sorted(validation),
    }


def counts(splits: dict[str, Any], tasks: list[Task]) -> dict[str, dict[str, int]]:
    cat = {t.id: t.category for t in tasks}
    out: dict[str, dict[str, int]] = {}
    for name in ("probe", "validation"):
        d: dict[str, int] = {}
        for i in splits[name]:
            d[cat[i]] = d.get(cat[i], 0) + 1
        out[name] = d
    return out


def load_splits(path: Path = SPLITS_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def select(tasks: list[Task], ids: list[str]) -> list[Task]:
    by_id = {t.id: t for t in tasks}
    return [by_id[i] for i in ids]


if __name__ == "__main__":
    all_tasks = load_all(CATEGORIES)
    sp = make_splits(all_tasks)
    SPLITS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SPLITS_PATH.write_text(json.dumps(sp, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(counts(sp, all_tasks), indent=1))
