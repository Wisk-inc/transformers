# Training Pluvium: commands you can copy

Pluvium has three things that are trained or learned. Each has one command.

| What | Needs | Time | Command |
|---|---|---|---|
| **Pluvium Blend** for a country: 14 days of daily rain chances from six operational systems | A CPU, internet, about 5 GB of disk | About 2 hours, mostly downloads | `pluvium blend train` |
| **Pluvium-500**, the hour-by-hour network | One GPU with 24 GB or more, about 150 GB of disk | 24 hours by default | `pluvium program` |
| **Pluvium Cyclone**, hurricane tracks | A CPU, internet | Minutes per run. Nothing to train: it starts from weights learned on 2024 and 2025 and keeps learning | `pluvium cyclone` |

## 1. Pluvium Blend for any country (CPU)

```bash
pip install --upgrade pluvium
pluvium setup --country JM --root ~/pluvium-JM          # area, terrain, towns (any ISO codes: TT, or DO,HT)
pluvium blend train --root ~/pluvium-JM                 # a year of archived runs and satellite rain, then the fit
pluvium blend forecast --root ~/pluvium-JM              # 14 days for every town -> blend_YYYY-MM-DD.csv
```

Every day after that:

```bash
pluvium blend learn --root ~/pluvium-JM                 # update the online weights from the days now known
pluvium blend forecast --root ~/pluvium-JM
```

`blend train` fits the design benchmarked in [FRONTIER_BENCHMARK.md](FRONTIER_BENCHMARK.md): components trained on
every heavy day, the pool, the online weights, and the record-storm hand-over. `--classic` trains the first Blend
instead. `--start` and `--end` change the training period.

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
Hub, and makes a live forecast. Running the cell again continues from the last checkpoint.

The file [notebooks/rtx6000_pluvium.py](../notebooks/rtx6000_pluvium.py) is the same cell with a fallback that
installs from source.

## 3. Pluvium Cyclone (CPU)

```bash
pluvium cyclone --root ~/storms                          # every storm alive now: track, cone, passage chances
pluvium cyclone --root ~/storms --time 2025-10-25T22     # as it would have been issued then
```

It downloads the season's decks from the National Hurricane Center and sea-level pressure from six systems for the
last ten days of runs, tracks every storm in every member, and writes a bulletin and a data file per storm.

## 4. Putting the code and papers on the Hub

```bash
python hub/upload_to_hub.py            # after putting your token in HF_TOKEN at the top of the file
```

That creates the model repository if needed and uploads the model card, every source file the architecture is
built from, the tests, the papers and the benchmarks, in one commit.

## 5. Repeating the benchmarks

```bash
python -m pluvium.benchmarks.fullyear --archive ARCHIVE --truth TRUTH --out OUT        # the first full-year benchmark
python -m pluvium.benchmarks.cyclone_bench --decks DECKS --fields PRESSURE --out OUT   # hurricanes
```

The frontier benchmark builds a store per place (`pluvium.frontier.store.build_store`), scores designs on it
(`pluvium.frontier.evaluate.evaluate`), and writes its page with `scripts/doc_frontier.py`.
