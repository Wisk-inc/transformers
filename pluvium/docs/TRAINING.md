# Training Pluvium: commands you can copy

Pluvium has three things that are trained or learned. Each has one command.

| What | Needs | Time | Command |
|---|---|---|---|
| Pluvium Blend for a country: 14 days of daily rain chances from six operational systems | A CPU, internet, 41 MB of disk for Jamaica and a few GB for a large country | About an hour and a half, mostly downloads (86 minutes for Jamaica on four cores) | `pluvium now`, or `pluvium blend train` |
| Pluvium-500, the hour-by-hour network | One GPU with 24 GB or more, about 150 GB of disk | 24 hours by default | `pluvium program` |
| Pluvium Cyclone, hurricane tracks | A CPU, internet | Minutes per run. Nothing to train: it starts from weights learned on 2024 and 2025 and keeps learning | `pluvium cyclone` |

One notebook cell does all three and builds the Hub repository as well:
[notebooks/pluvium_everything.py](../notebooks/pluvium_everything.py). It needs nothing changed to forecast the
rain for Jamaica on a CPU. With a Hugging Face write token pasted in it also uploads the code, the papers and the
upstream weights, and on a GPU of 24 GB or more it also trains the network.

## 1. Pluvium Blend for any country (CPU)

The short way:

```bash
pip install --upgrade pluvium
pluvium now --country JM                                # sets up, downloads or trains a blend, forecasts
```

Step by step:

```bash
pluvium setup --country JM --root ~/pluvium-JM          # area, terrain, towns (any ISO codes: TT, or DO,HT)
pluvium blend train --root ~/pluvium-JM                 # a year of archived runs and satellite rain, then the fit
pluvium blend forecast --root ~/pluvium-JM              # 14 days for every town -> blend_YYYY-MM-DD.csv
pluvium blend learn --root ~/pluvium-JM                 # update the online weights from rain published since
pluvium blend publish --root ~/pluvium-JM --repo you/your-repo    # share the blend (needs HF_TOKEN)
```

`blend train` fits the design benchmarked in [FRONTIER_BENCHMARK.md](FRONTIER_BENCHMARK.md): components trained on
every heavy day, the pool, the online weights, and the record-storm hand-over. At the end it stages record storms:
the four wettest days of the training period are withheld, forecast without having been seen, and scored with and
without the hand-over. It prints the verdict: kept, or switched off because those storms were not forecast better
with it. `--handover always` keeps it whatever the verdict, and `--handover never` drops it. The verdict rests on
four storms and can be wrong. A country in the path of hurricanes may prefer to hand every record forecast to the
raw systems, which is what `always` is for. `--classic` trains the first Blend instead.

The training period is chosen for you. A blend learns from the rain that fell after each forecast, and NOAA
publishes the satellite rain record (CMORPH) about four and a half months late, so the period is the most recent
year that record covers; in November 2026 that was 16 May 2025 to 16 May 2026. `--start` and `--end` set another
period. If the record does not cover it, the command stops and names a period that will work.

For the same reason `blend learn` has nothing to learn from on most days and says so. When NOAA publishes another
month, each call works through two weeks of it, oldest day first.

The forecast table has one row per place and day: the chance of 1, 10, 25 and 50 mm, the median, 90th percentile and
expected rain, how many raw systems call for 50 mm, the alert tier (watch, warning, extreme), whether the forecast is
beyond the training record, and how rare the expected amount is at that place.

The step-by-step commands as one cell for a notebook, with hurricanes included:
[notebooks/pluvium_blend_cell.py](../notebooks/pluvium_blend_cell.py).

```python
COUNTRY = "JM"                              # ISO codes: "JM", "TT", "DO,HT", "PH", "KE" ...
ROOT = "~/pluvium-JM"                       # working folder; 41 MB for Jamaica, a few GB for a large country

import os, subprocess, sys

def sh(cmd, check=True):
    print(f"$ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, check=check).returncode

sh(f"{sys.executable} -m pip install -q --upgrade 'pluvium>=0.4.2'")
root = os.path.expanduser(ROOT)
pluvium = f"{sys.executable} -m pluvium.cli"
if not os.path.exists(f"{root}/area.json"):
    sh(f"{pluvium} setup --country {COUNTRY} --root {root}")
if not os.path.exists(f"{root}/blend/blend.json"):
    sh(f"{pluvium} blend train --root {root}")              # about 90 minutes the first time, mostly downloads
sh(f"{pluvium} blend learn --root {root}", check=False)     # learns only when NOAA has published more rain
sh(f"{pluvium} blend forecast --root {root}")
sh(f"{pluvium} cyclone --root {root}", check=False)         # every tropical cyclone alive now
```

## 2. Pluvium-500, the network (GPU)

Paste this into one notebook cell on a GPU machine, or save it as a file and run it with Python. Put a Hugging Face
write token where it says so: checkpoints and data are mirrored to the Hub, so a session that stops can pick up where
it left off, and the finished model is published there.

```python
GPU_HOURS = 24                              # wall-clock budget for this session
HF_TOKEN = "PASTE_YOUR_ORG_TOKEN_HERE"      # a write token (Hub -> Settings -> Access Tokens)
HF_REPO = "CorxLabs/Pluvium-500"            # where data, checkpoints and the finished model are kept
ROOT = "/workspace/pluvium"                 # working folder with about 150 GB free
SIZE = "pluvium-500"                        # or pluvium-base for a smaller, faster model
COUNTRY = ""                                # ISO codes such as "JM" or "DO,HT"; empty for the whole Caribbean

import os, subprocess, sys

def sh(cmd):
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)

sh(f"{sys.executable} -m pip install -q --upgrade pluvium")
probe = ("import torch; assert torch.cuda.is_available(), 'no GPU visible';"
         "x = torch.ones(8, device='cuda'); print(torch.cuda.get_device_name(0), torch.__version__, float(x @ x))")
if subprocess.run([sys.executable, "-c", probe]).returncode != 0:
    sh(f"{sys.executable} -m pip install -q --upgrade torch --index-url https://download.pytorch.org/whl/cu128")
    subprocess.run([sys.executable, "-c", probe], check=True)

env = dict(os.environ, GPU_HOURS=str(GPU_HOURS), HF_REPO=HF_REPO, PLUVIUM_ROOT=ROOT, PLUVIUM_SIZE=SIZE,
           PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", TOKENIZERS_PARALLELISM="false")
if not HF_TOKEN.startswith("PASTE"):
    env["HF_TOKEN"] = HF_TOKEN
else:
    print("No Hub token: nothing is mirrored, so a stopped session cannot resume and nothing is published.")
cmd = [sys.executable, "-u", "-m", "pluvium.program"] + (["--country", COUNTRY] if COUNTRY else [])
proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
try:
    for line in proc.stdout:
        print(line, end="", flush=True)
finally:
    if proc.poll() is None:
        proc.terminate()
print(f"finished with exit code {proc.wait()}; the forecast is in {ROOT}/forecast/report.txt")
```

The same thing as one shell line:

```bash
GPU_HOURS=24 HF_TOKEN=your_token HF_REPO=CorxLabs/Pluvium-500 pluvium program
```

What it does, in order: reads the area, stages ERA5, CMORPH and airport reports, computes the climatology, trains on
reanalysis (stage A), trains on real GFS runs (stage B), calibrates and scores on a held-out year, publishes to the
Hub, and makes a live forecast. Running the cell again continues from the last checkpoint. The network's own card,
with the scores measured in the run, is published as `NETWORK_CARD.md`; it becomes the repository's front page only
if the repository has none yet.

The file [notebooks/rtx6000_pluvium.py](../notebooks/rtx6000_pluvium.py) is the same cell with a fallback that
installs from source.

## 3. Pluvium Cyclone (CPU)

```bash
pluvium cyclone --root ~/storms                          # every storm alive now: track, cone, passage chances
pluvium cyclone --root ~/storms --time 2025-10-25T22     # as it would have been issued then
```

It downloads the season's decks from the National Hurricane Center and sea-level pressure from six systems for the
last ten days of runs, tracks every storm in every member, and writes a bulletin and a data file per storm.

## 4. The Hub repository

```bash
python hub/upload_to_hub.py            # after putting your token in HF_TOKEN at the top of the file
HF_TOKEN=your_token pluvium upstream mirror --repo CorxLabs/Pluvium-500
HF_TOKEN=your_token pluvium blend publish --root ~/pluvium-JM --repo CorxLabs/Pluvium-500
```

The first command creates the model repository if needed and uploads the model card, every source file the
architecture is built from, the tests, the papers and the benchmarks, in one commit. It also creates a collection
in the organisation that holds the model and the papers it builds on.

The second copies the published weights of the AI models whose forecasts Pluvium reads (DeepMind's GraphCast,
NOAA's GraphCastGFS and AIGFS, ECMWF's AIFS) into `upstream/`, unchanged, each with its licence. They come to 4.9 GB.
A model that is already there at the same sizes is skipped. `pluvium upstream list` shows what would be copied.

The third uploads a trained blend to `blends/<country>/`, after which `pluvium now` downloads it for anybody who
asks for that country.

## 5. Repeating the benchmarks

```bash
python -m pluvium.benchmarks.fullyear --archive ARCHIVE --truth TRUTH --out OUT        # the first full-year benchmark
python -m pluvium.benchmarks.frontier_bench --archive ARCHIVE --truth TRUTH --out OUT  # the second
python -m pluvium.benchmarks.cyclone_bench --decks DECKS --fields PRESSURE --out OUT   # hurricanes
```

The frontier benchmark builds a store per place (`pluvium.frontier.store.build_store`, about an hour per place on
four cores, resumable) and scores every design on it (`pluvium.frontier.evaluate.evaluate`; its `latency` argument
is the delay, in hours, before a day's rain is known to the online weights). Its page, the papers and the charts
are then written from the result files, so every number in them is the data's:

```bash
python scripts/doc_frontier.py OUT/results FIRST EXPERIMENTS docs/FRONTIER_BENCHMARK.md
python scripts/doc_papers.py OUT/results FIRST EXPERIMENTS docs/papers
python scripts/chart_frontier.py OUT/results FIRST docs/img
python scripts/check_style.py          # the papers, the card and the readme against the style check
```

`FIRST` is the folder of the first benchmark's results and `EXPERIMENTS` the folder of withheld-storm results
(`pluvium.tails.withheld.withheld_experiment`, one JSON file per place). A folder `results_late140` beside
`OUT/results`, holding the same evaluation run with `latency=24 * 140`, adds the tables on the late rain record.
