# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
The trainer on the tiny corpus: it trains, logs, checkpoints, exports, resumes to the same numbers, and
runs sharded over two processes with FSDP2 and distributed checkpoints (on CPU here, the same code on GPUs).
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import torch

from helix_lm import HelixForCausalLM
from helix_lm.train import Trainer, parse_args

SMALL = [
    "--size", "20m", "--seq-len", "32", "--micro-batch", "2", "--global-batch-tokens", "128",
    "--loss-chunk-tokens", "16", "--log-every", "1", "--eval-every", "0", "--eval-batches", "1",
    "--warmup-steps", "2", "--lr", "1e-3", "--holdout-tokens", "100", "--push-every-min", "1e9",
    "--override", "block_size=8", "--override", "local_blocks=2", "--override", "index_branching=2",
    "--override", "index_topk=2", "--override", "landmark_dim=16", "--override", "attention_backend=sdpa",
]  # fmt: skip


def args_for(prepared, out, *extra):
    tokenizer = prepared.parent / "tok"
    return parse_args(["--data", str(prepared), "--out", str(out), "--tokenizer", str(tokenizer), *SMALL, *extra])


def read_log(out: Path, event: str) -> list[dict]:
    lines = [json.loads(line) for line in (out / "log.jsonl").read_text().splitlines()]
    return [r for r in lines if r.get("event") == event]


def test_single_process_trains_checkpoints_and_exports(prepared, tmp_path):
    out = tmp_path / "run"
    trainer = Trainer(args_for(prepared, out, "--steps", "6", "--save-every-steps", "3"))
    assert trainer.accumulation == 2 and trainer.tokens_per_step == 128
    assert trainer.run() == "planned"
    steps = read_log(out, "step")
    assert [s["step"] for s in steps] == [1, 2, 3, 4, 5, 6]
    assert all(torch.isfinite(torch.tensor(s["loss"])) for s in steps)
    assert steps[0]["lr"] < steps[1]["lr"], "warmup ramps the learning rate"
    assert steps[-1]["lr"] < steps[2]["lr"], "the last 20% decays"
    assert read_log(out, "finished")[0]["tokens"] == 6 * 128
    assert (out / "checkpoints" / "latest").read_text() == "step-00000006"
    assert sorted(p.name for p in (out / "checkpoints").glob("step-*")) == ["step-00000003", "step-00000006"]
    export = out / "export"
    assert (export / "model.safetensors").exists() or (export / "model.safetensors.index.json").exists()
    assert (export / "tokenizer.json").exists() and json.loads((export / "training_state.json").read_text())[
        "step"
    ] == 6
    loaded = HelixForCausalLM.from_pretrained(export)
    assert loaded.num_parameters() == trainer.config.num_parameters()
    from safetensors import safe_open

    weights = next(iter(sorted(export.glob("model*.safetensors"))))
    with safe_open(weights, "pt") as handle:
        assert handle.get_tensor(next(iter(handle.keys()))).dtype == torch.bfloat16, "the export is bf16"


def test_resume_reproduces_the_uninterrupted_run(prepared, tmp_path):
    straight = tmp_path / "straight"
    Trainer(args_for(prepared, straight, "--steps", "6", "--save-every-steps", "100")).run()
    # The same 6-step plan, killed right after its step-3 checkpoint: the stop file appears when that save lands.
    broken, stop = tmp_path / "broken", tmp_path / "kill.now"
    interrupted = Trainer(
        args_for(prepared, broken, "--steps", "6", "--save-every-steps", "3", "--stop-file", str(stop))
    )
    original = interrupted.save_checkpoint

    def save_then_die():
        original()
        stop.write_text("now")

    interrupted.save_checkpoint = save_then_die
    assert interrupted.run() == "stop-file" and interrupted.step == 3
    resumed = Trainer(args_for(prepared, broken, "--steps", "6", "--save-every-steps", "100"))
    assert resumed.step == 3 and resumed.tokens_seen == 3 * 128
    resumed.run()
    a, b = read_log(straight, "step"), read_log(broken, "step")
    assert [r["step"] for r in b] == [1, 2, 3, 4, 5, 6]
    for x, y in zip(a[3:], [r for r in b if r["step"] > 3], strict=True):
        assert abs(x["loss"] - y["loss"]) < 1e-5, (x["step"], x["loss"], y["loss"])
    first = HelixForCausalLM.from_pretrained(straight / "export")
    second = HelixForCausalLM.from_pretrained(broken / "export")
    for (name, p), (_, q) in zip(first.named_parameters(), second.named_parameters(), strict=True):
        torch.testing.assert_close(q, p, msg=name)


def test_stop_file_ends_the_run_after_a_save(prepared, tmp_path):
    out = tmp_path / "stop"
    stop = tmp_path / "stop.now"
    stop.write_text("stop")
    trainer = Trainer(args_for(prepared, out, "--steps", "50", "--stop-file", str(stop)))
    assert trainer.run() == "stop-file"
    assert trainer.step == 0 and (out / "checkpoints" / "latest").exists()


def test_hours_schedule_is_fitted_from_measured_throughput(prepared, tmp_path):
    out = tmp_path / "hours"
    trainer = Trainer(args_for(prepared, out, "--hours", "0.02", "--measure-steps", "3"))
    assert trainer.total_steps is None
    # The test cannot wait 72 seconds, nor keep minutes in reserve: end 25 s from the start with tiny margins.
    trainer.deadline = trainer.started + 25
    trainer.stop_margin, trainer.reserve = 0.5, 1.0
    trainer.run()
    assert trainer.total_steps is not None and trainer.step >= 3
    finished = read_log(out, "finished")[0]
    assert finished["reason"] in ("planned", "deadline")


def test_init_from_starts_sft_from_a_saved_model(prepared, tmp_path):
    base = tmp_path / "base"
    Trainer(args_for(prepared, base, "--steps", "2")).run()
    sft = tmp_path / "sft"
    trainer = Trainer(args_for(prepared, sft, "--steps", "1", "--init-from", str(base / "export")))
    pretrained = HelixForCausalLM.from_pretrained(base / "export")
    for (name, p), (_, q) in zip(pretrained.named_parameters(), trainer.model.named_parameters(), strict=True):
        torch.testing.assert_close(q.float(), p.float(), msg=name)


def test_parse_args_requires_a_length(prepared, tmp_path):
    with pytest.raises(SystemExit):
        parse_args(["--data", str(prepared), "--out", str(tmp_path)])


@pytest.mark.skipif(shutil.which("torchrun") is None, reason="torchrun not on PATH")
def test_two_processes_fsdp2_train_checkpoint_and_resume(prepared, tmp_path):
    out = tmp_path / "fsdp"
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"), "OMP_NUM_THREADS": "1"}

    def run(*extra) -> subprocess.CompletedProcess:
        command = [
            "torchrun", "--nproc_per_node=2", "--master_port=29517", "-m", "helix_lm.train", "--data", str(prepared),
            "--out", str(out), "--tokenizer", str(prepared.parent / "tok"), "--parallel", "fsdp", *SMALL, *extra,
        ]  # fmt: skip
        return subprocess.run(command, env=env, capture_output=True, text=True, timeout=600)

    first = run("--steps", "4", "--save-every-steps", "2")
    assert first.returncode == 0, first.stdout[-3000:] + first.stderr[-3000:]
    assert "fsdp" in first.stdout and "[train] checkpoint at step 4" in first.stdout
    assert (out / "checkpoints" / "step-00000004" / "dcp").exists()
    second = run("--steps", "6", "--save-every-steps", "100")
    assert second.returncode == 0, second.stdout[-3000:] + second.stderr[-3000:]
    assert "resumed from step-00000004" in second.stdout
    steps = read_log(out, "step")
    assert [s["step"] for s in steps] == [1, 2, 3, 4, 5, 6]
    assert HelixForCausalLM.from_pretrained(out / "export").num_parameters() > 0


ANANSI = [
    "--size", "a-20m", "--override", "ngram_memory_size=512", "--override", "num_streams=2",
    "--override", "mean_loops=2", "--override", "max_loops=3", "--override", "backprop_loops=2",
]  # fmt: skip


def test_anansi_trains_with_muon_and_mup_and_resumes(prepared, tmp_path):
    """The 2.0 path: Anansi with Muon + muP, the multi-token loss logged, checkpoint and resume with Chained state."""
    out = tmp_path / "anansi"
    base = [*ANANSI, "--optimizer", "muon", "--mup", "--save-every-steps", "3"]
    first = Trainer(args_for(prepared, out, *base, "--steps", "3"))
    assert first.config.output_multiplier == 256 / first.config.hidden_size
    assert first.mup is not None and len(first.optimizer.optimizers) == 2
    assert first.run() == "planned"
    steps = read_log(out, "step")
    assert [s["step"] for s in steps] == [1, 2, 3] and all(s.get("loops", 0) >= 1 for s in steps)
    assert all(s.get("mtp", 0) > 0 for s in steps), "the multi-token loss is logged"
    second = Trainer(args_for(prepared, out, *base, "--steps", "6"))
    assert second.step == 3, "resumed from the checkpoint"
    assert second.run() == "planned"
    assert [s["step"] for s in read_log(out, "step")] == [1, 2, 3, 4, 5, 6]
    from helix_lm.anansi import AnansiForCausalLM

    loaded = AnansiForCausalLM.from_pretrained(out / "export")
    assert loaded.config.num_streams == 2 and loaded.num_parameters() == second.config.num_parameters()


def test_online_reweighting_runs_in_the_trainer(prepared, tmp_path):
    out = tmp_path / "reweight"
    trainer = Trainer(args_for(prepared, out, "--steps", "6", "--eval-every", "2", "--reweight-strength", "1.0"))
    assert trainer.run() == "planned"
    assert len(read_log(out, "eval")) >= 3
    weights = trainer.loader.weights
    assert abs(sum(weights.values()) - 1.0) < 1e-6 or sum(weights.values()) > 0
    prior = trainer.loader.prior
    assert all(0.5 * prior[n] - 1e-9 <= w <= 2.0 * prior[n] + 1e-9 for n, w in weights.items())
