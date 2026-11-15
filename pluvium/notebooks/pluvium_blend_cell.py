# Pluvium Blend and Pluvium Cyclone for any country, on a CPU: set up, train, forecast.
# Paste into one notebook cell (Colab, Kaggle, Jupyter, a laptop) or run as `python pluvium_blend_cell.py`.
# No GPU and no account are needed. Running it again reuses everything already downloaded.

COUNTRY = "JM"                              # ISO codes: "JM", "TT", "DO,HT", "PH", "KE" ...
ROOT = "~/pluvium-JM"                       # working folder; needs about 5 GB
TRAIN_START = "2024-04-01T00"               # the training period: a year of archived runs and satellite rain
TRAIN_END = "2025-04-01T00"
HURRICANES = True                           # also write a bulletin for every tropical cyclone alive now
BASINS = "al,ep"                            # al: North Atlantic, ep: East Pacific

import os
import subprocess
import sys
from pathlib import Path


def sh(cmd: str, check: bool = True) -> int:
    print(f"$ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, check=check).returncode


# 1. the package: from PyPI, or from the source tree when PyPI is not reachable
if sh(f"{sys.executable} -m pip install -q --upgrade 'pluvium>=0.3.0'", check=False) != 0:
    sh(f"{sys.executable} -m pip install -q 'git+https://github.com/Wisk-inc/transformers.git"
       f"@claude/linear-context-architecture-a35c6h#subdirectory=pluvium'")

root = Path(os.path.expanduser(ROOT))
pluvium = f"{sys.executable} -m pluvium.cli"

# 2. the area: borders, terrain and every town of the countries asked for
if not (root / "area.json").exists():
    sh(f"{pluvium} setup --country {COUNTRY} --root {root}")

# 3. the blend: about two hours the first time, nearly all of it downloads. It prints the hand-over audit's verdict
#    (whether record storms are handed to the raw systems here) and saves the model in ROOT/blend.
if not (root / "blend" / "blend.json").exists():
    sh(f"{pluvium} blend train --root {root} --start {TRAIN_START} --end {TRAIN_END}")

# 4. every day: learn from the days whose rain is now known, then forecast 14 days for every town
sh(f"{pluvium} blend learn --root {root}", check=False)        # nothing to learn on the first day: that is fine
sh(f"{pluvium} blend forecast --root {root}")

# 5. hurricanes: track, cone, chance of passing near each place, for every storm alive now
if HURRICANES:
    sh(f"{pluvium} cyclone --root {root} --basins {BASINS}", check=False)

tables = sorted(root.glob("blend_*.csv"))
print(f"\nforecast table: {tables[-1] if tables else 'none written'}")
print("columns: chance of 1, 10, 25 and 50 mm; median, 90th percentile and expected rain; how many raw systems call "
      "for 50 mm; the alert tier; whether the forecast is beyond the training record; how rare the expected amount is")
if HURRICANES:
    print(f"hurricane bulletins and data: {root / 'cyclone'}")
