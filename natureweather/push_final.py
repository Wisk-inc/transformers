# NatureV1 -- push everything the finished run left only on this machine to the Hub, then verify it.
#
# Paste into a new molab cell and run once, in the same session as the finished run (the files live on
# this session's disk). Each file is compared with the Hub copy by SHA-256, not by size: the storm
# banks were rebuilt at exactly the same size, which is how the size check let stale copies stand.
import hashlib, json, os, time
from pathlib import Path

REPO = "Sigmandndnns/NatureV1-500"
DATA = Path("/home/marimo/naturev1_data")
CKPT = DATA / "naturev1_ckpt"

from huggingface_hub import HfApi, get_token
from huggingface_hub.utils import disable_progress_bars

disable_progress_bars()
token = os.environ.get("HF_TOKEN") or get_token()
if not token:
    raise SystemExit("no Hugging Face token in this session: set HF_TOKEN (os.environ['HF_TOKEN'] = '...') "
                     "in a cell, run this again, then delete that line")
api = HfApi(token=token)

files = {                                              # local file -> path in the repo
    CKPT / "stage1" / "latest.pt": "checkpoints/stage1/latest.pt",
    CKPT / "rollout" / "latest.pt": "checkpoints/rollout/latest.pt",
    CKPT / "stage2" / "latest.pt": "checkpoints/stage2/latest.pt",
    CKPT / "storm_bank_train.pt": "checkpoints/storm_bank_train.pt",
    CKPT / "storm_bank_val.pt": "checkpoints/storm_bank_val.pt",
    CKPT / "batch.json": "staging/batch.json",
    DATA / "stats.json": "staging/stats.json",
    DATA / "train.log": "logs/train.log",
}
for tag in ("stage1", "rollout", "stage2"):
    files[CKPT / tag / "state.json"] = f"checkpoints/{tag}/state.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def remote_sha(repo_path: str):
    try:
        for entry in api.get_paths_info(REPO, [repo_path], repo_type="model"):
            lfs = getattr(entry, "lfs", None)
            if lfs is not None:
                return lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
            # a small file is stored as plain git: its blob id is not a SHA-256, so compare the bytes
            local = api.hf_hub_download(REPO, repo_path, repo_type="model", force_download=True)
            return sha256(Path(local))
    except Exception:
        return None
    return None


def step_of(path: Path) -> str:
    state = path.parent / "state.json"
    try:
        return f"step {json.loads(state.read_text())['step']:,}"
    except Exception:
        return ""


print(f"{'file':<34} {'size':>9}  status")
for local, repo_path in files.items():
    if not local.exists():
        print(f"{repo_path:<34} {'':>9}  not on this machine -- skipped")
        continue
    size = local.stat().st_size
    mine = sha256(local)
    label = f"{repo_path:<34} {size / 1e9:>7.2f}GB"
    if remote_sha(repo_path) == mine:
        print(f"{label}  already identical on the Hub {step_of(local)}")
        continue
    began = time.time()
    api.upload_file(path_or_fileobj=str(local), path_in_repo=repo_path, repo_id=REPO, repo_type="model",
                    commit_message=f"final {repo_path} {step_of(local)}".strip())
    verified = remote_sha(repo_path) == mine
    print(f"{label}  pushed in {time.time() - began:.0f} s {step_of(local)} -- "
          + ("verified" if verified else "!! Hub copy does not match, run this cell again"))

print(f"\ndone: https://huggingface.co/{REPO}/tree/main")
