"""Download the BFCL v4 data files used by ToolQuant into data/cache/bfcl (or $BFCL_DATA_DIR).

Uses `pip download` of the public `bfcl-eval` wheel into a temp dir and copies only data files.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

FILES = [
    "BFCL_v4_simple_python.json",
    "BFCL_v4_multiple.json",
    "BFCL_v4_irrelevance.json",
]


def main() -> None:
    dest = Path(os.environ.get("BFCL_DATA_DIR") or "data/cache/bfcl")
    (dest / "possible_answer").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            [sys.executable, "-m", "pip", "download", "bfcl-eval", "--no-deps", "-d", tmp],
            check=True,
        )
        whl = next(Path(tmp).glob("bfcl_eval-*"))
        with zipfile.ZipFile(whl) as z:
            for f in FILES:
                for sub, out in (
                    (f"bfcl_eval/data/{f}", dest / f),
                    (f"bfcl_eval/data/possible_answer/{f}", dest / "possible_answer" / f),
                ):
                    try:
                        with z.open(sub) as src, open(out, "wb") as dst:
                            shutil.copyfileobj(src, dst)
                    except KeyError:
                        pass  # irrelevance has no possible_answer file
    print("wrote", dest)


if __name__ == "__main__":
    main()
