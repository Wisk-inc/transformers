# NatureV1 -- publish the finished model to the Hub: everything the run left on this machine, plus the release.
#
# Paste into a new molab cell, put your Hugging Face WRITE token in _HF_TOKEN below, and run it once -- in the
# session that finished the run if you can (anything not on this disk is taken from the Hub instead).
# Afterwards, clear _HF_TOKEN back to "" so the token is not saved in the notebook or shared with it.
# Re-running is safe: anything already identical on the Hub is skipped. Everything runs inside one function,
# so this cell defines nothing that clashes with the training cell (marimo refuses cells that redefine names).
_HF_TOKEN = ""         # <- paste your token between the quotes: "hf_..."  (huggingface.co/settings/tokens, Write)


def _publish_naturev1(hf_token: str) -> None:
    import hashlib, json, os, subprocess, sys, tempfile, time, urllib.request
    from dataclasses import asdict
    from pathlib import Path

    REPO = "CorxLabs/NatureV1-500"
    DATA = Path("/home/marimo/naturev1_data")
    CKPT = DATA / "naturev1_ckpt"
    SOURCE = ("https://raw.githubusercontent.com/Wisk-inc/transformers/"
              "claude/linear-context-architecture-a35c6h/natureweather")
    CARD = "HF_README.md"
    RESULTS = ["elnino_2023_end_to_end.txt", "heldout_landfall_ri_structure.txt",
               "heldout_per_storm_and_hindcasts.txt", "heldout_storm_scorecard_2017_2019_2021.txt",
               "maria_2017_end_to_end.txt", "redteam_ablation.txt", "redteam_calibration.txt",
               "redteam_consistency_and_leakage.txt", "redteam_otis_2023.txt", "redteam_stress.txt"]
    EVAL = ["README.md", "arco15.py", "e2e_compare.py", "e2e_maria.py", "elnino2023.py", "other_heads.py",
            "per_storm.py", "score2023.py", "test_storms.py", "validate15.py", "stress.py", "calib.py", "ablate.py",
            "consistency.py", "ood_run.py", "score_ood.py"]

    try:
        import safetensors  # noqa: F401
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "safetensors"], check=True)
    import torch
    from huggingface_hub import HfApi, get_token, hf_hub_download
    from huggingface_hub.utils import disable_progress_bars
    from safetensors.torch import save_file

    disable_progress_bars()
    token = hf_token.strip() or os.environ.get("HF_TOKEN") or get_token()
    if not token:
        raise SystemExit("no Hugging Face token: paste your write token into _HF_TOKEN at the top of this cell")
    api = HfApi(token=token)


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
                with tempfile.TemporaryDirectory() as scratch:   # small files are plain git: compare the bytes
                    return sha256(Path(hf_hub_download(REPO, repo_path, repo_type="model", local_dir=scratch,
                                                       token=token)))
        except Exception:
            return None
        return None


    def tensor_digest(tensors: dict) -> str:
        """A hash of the weights themselves: safetensors writes its header in a different order in every process."""
        digest = hashlib.sha256()
        for name in sorted(tensors):
            value = tensors[name]
            digest.update(f"{name}|{value.dtype}|{tuple(value.shape)}|".encode())
            digest.update(value.contiguous().view(-1).view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()


    def remote_tensor_digest(repo_path: str):
        try:
            from huggingface_hub import get_safetensors_metadata

            found = get_safetensors_metadata(REPO, token=token).files_metadata.get(repo_path)
            return (found.metadata or {}).get("tensor_sha256") if found is not None else None
        except Exception:
            return None


    def publish(local: Path, repo_path: str, note: str = "", digest: str | None = None) -> None:
        label = f"{repo_path:<46} {local.stat().st_size / 1e6:>9.1f} MB"
        if digest is not None and remote_tensor_digest(repo_path) == digest:
            print(f"{label}  already identical weights {note}")
            return
        mine = sha256(local)
        if remote_sha(repo_path) == mine:
            print(f"{label}  already identical {note}")
            return
        began = time.time()
        api.upload_file(path_or_fileobj=str(local), path_in_repo=repo_path, repo_id=REPO, repo_type="model",
                        commit_message=f"{repo_path} {note}".strip())
        ok = remote_sha(repo_path) == mine
        print(f"{label}  pushed in {time.time() - began:.0f} s {note} -- "
              + ("verified" if ok else "!! Hub copy does not match: run this cell again"))


    def step_of(directory: Path) -> str:
        try:
            return f"(step {json.loads((directory / 'state.json').read_text())['step']:,})"
        except Exception:
            return ""


    # 1 ── what the run left on this disk ───────────────────────────────────────────────────────────────────────
    print("1. training files")
    files = {CKPT / "storm_bank_train.pt": "checkpoints/storm_bank_train.pt",
             CKPT / "storm_bank_val.pt": "checkpoints/storm_bank_val.pt",
             CKPT / "satellite_bank_train.pt": "checkpoints/satellite_bank_train.pt",
             CKPT / "satellite_bank_val.pt": "checkpoints/satellite_bank_val.pt",
             CKPT / "batch.json": "staging/batch.json", DATA / "stats.json": "staging/stats.json",
             DATA / "train.log": "logs/train.log"}
    for tag in ("stage1", "rollout", "stage2", "stage3"):
        files[CKPT / tag / "latest.pt"] = f"checkpoints/{tag}/latest.pt"
        files[CKPT / tag / "state.json"] = f"checkpoints/{tag}/state.json"
    for local, repo_path in files.items():
        if local.exists():
            publish(local, repo_path, step_of(local.parent) if local.name in ("latest.pt", "state.json") else "")
        else:
            print(f"{repo_path:<46} {'':>12}  not on this machine -- the Hub copy stays")

    # 2 ── the release, from the final model ────────────────────────────────────────────────────────────────────
    print("\n2. release")
    from ihelix import fibonacci_sphere
    from naturev1.model import NatureConfig, NatureV1

    # The released model is the last stage that exists: stage three (satellite, calibration, input monitor)
    # once it has run, otherwise stage two.
    final = next((CKPT / tag / "latest.pt" for tag in ("stage3", "stage2") if (CKPT / tag / "latest.pt").exists()),
                 None)
    if final is None:
        for tag in ("stage3", "stage2"):
            try:
                final = Path(hf_hub_download(REPO, f"checkpoints/{tag}/latest.pt", repo_type="model", token=token))
                break
            except Exception:
                continue
    payload = torch.load(final, map_location="cpu", weights_only=False)
    config = NatureConfig.from_dict(payload["config"])
    model = NatureV1(config, fibonacci_sphere(config.latent_points, num_neighbours=config.latent_neighbours,
                                              cluster_size=config.latent_cluster))
    model.load_state_dict(payload["model"])
    model.eval()
    stats = DATA / "stats.json"
    if not stats.exists():
        stats = Path(hf_hub_download(REPO, "staging/stats.json", repo_type="model", token=token))

    out = DATA / "hf_release"
    (out / "results").mkdir(parents=True, exist_ok=True)
    (out / "eval").mkdir(parents=True, exist_ok=True)
    state = {key: value.detach().contiguous().clone() for key, value in model.state_dict().items()}
    weights_digest = tensor_digest(state)
    save_file(state, out / "model.safetensors",
              metadata={"format": "pt", "model": "NatureV1", "tensor_sha256": weights_digest})
    torch.save(model.state_dict(), out / "model.safetensors.pt")
    (out / "config.json").write_text(json.dumps(asdict(config), indent=2, default=list))
    (out / "normalizer.json").write_text(stats.read_text())
    (out / "training_record.json").write_text(json.dumps(
        {"trained_steps_per_output": model.trained_heads(), "released_from": final.parent.name,
             "final_stage_state": payload["state"],
         "parameters": model.num_parameters()}, indent=2, default=str))
    print(f"   NatureV1, {model.num_parameters():,} parameters, from {final.name} {step_of(final.parent)}")

    # 3 ── card, results and evaluation scripts from GitHub ─────────────────────────────────────────────────────
    for remote, local in ([(CARD, out / "README.md")]
                          + [(f"release/results/{n}", out / "results" / n) for n in RESULTS]
                          + [(f"release/eval/{n}", out / "eval" / n) for n in EVAL]):
        with urllib.request.urlopen(f"{SOURCE}/{remote}", timeout=60) as response:
            local.write_bytes(response.read())

    for local in sorted(p for p in out.rglob("*") if p.is_file()):
        publish(local, str(local.relative_to(out)), "release",
                digest=weights_digest if local.name == "model.safetensors" else None)

    # check: what anyone downloading it gets
    from naturev1.hub import from_pretrained

    check, _, normalizer = from_pretrained(REPO, device="cpu", token=token)
    same = all(torch.equal(check.state_dict()[k], v) for k, v in model.state_dict().items())
    print(f"\nfrom_pretrained('{REPO}'): {check.num_parameters():,} parameters, "
          f"{len(normalizer.variables) if normalizer else 0} normalised channels, "
          + ("weights identical to the trained model" if same else "!! weights differ -- run this cell again"))
    print(f"done: https://huggingface.co/{REPO}")


_publish_naturev1(_HF_TOKEN)
