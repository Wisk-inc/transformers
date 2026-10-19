# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
Measuring Helix against the models it is meant to compete with.

Two harnesses, one table. ``lm-evaluation-harness`` runs the knowledge and reasoning suites (MMLU-Pro,
GSM8K, ARC, HellaSwag, HumanEval ...) through :class:`HelixLM`, an adapter that gives it log-likelihoods
and greedy generations from a saved model. ``EvalPlus`` runs HumanEval+ and MBPP+: the model writes the
solutions here, EvalPlus executes the tests. ``compare`` prints both next to the published numbers of the
targets, so a result is always read against something.

The targets' figures are vendor-reported, from their model cards; a fair comparison re-runs them with the
same harness and settings, which ``helix eval --baseline Qwen/Qwen3.5-27B`` does when that model fits on
the machine (it is loaded through ``transformers``).
"""

from __future__ import annotations

import contextlib
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch

from .model import HelixForCausalLM
from .tokenizer import HelixTokenizer

# Published numbers (model cards; vendor-reported, see the docstring). Scores are percentages.
PUBLISHED = {
    "Qwen3.5-27B": {"mmlu_pro": 86.1, "livecodebench_v6": 80.7, "swe_bench_verified": 72.4},
    "Qwen2.5-Coder-32B-Instruct": {"humaneval": 92.7, "mbpp": 90.2, "humaneval_plus": 87.2, "mbpp_plus": 75.1},
    "Qwen2.5-7B": {"mmlu_pro": 56.3, "gsm8k": 85.4, "humaneval": 57.9, "mbpp": 74.9},
}

DEFAULT_TASKS = ("hellaswag", "arc_challenge", "gsm8k", "mmlu_pro", "humaneval")


try:
    from lm_eval.api.model import LM as _LMBase
except ImportError:  # pragma: no cover - lm_eval is an optional extra
    _LMBase = object


class HelixLM(_LMBase):
    """
    The adapter ``lm_eval`` drives. Log-likelihoods come from one teacher-forced forward per request (right
    padded, which HELIX handles exactly); generations are greedy, cut at the task's stop strings.
    """

    def __init__(self, model_dir: str, device: str | None = None, batch_size: int = 8, max_length: int = 8192) -> None:
        if _LMBase is not object:
            super().__init__()
        self._device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = HelixForCausalLM.from_pretrained(model_dir).to(self._device).eval()
        if self._device.type == "cuda":
            self.model = self.model.to(torch.bfloat16)
        self.tokenizer = HelixTokenizer.from_pretrained(model_dir)
        self.batch_size_value = batch_size
        self.max_length = max_length

    @property
    def device(self) -> torch.device:
        return self._device

    # --- what lm_eval asks for ------------------------------------------------------------------------
    @property
    def eot_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    @property
    def batch_size(self) -> int:
        return self.batch_size_value

    def tok_encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def tok_decode(self, ids) -> str:
        return self.tokenizer.decode(ids)

    def apply_chat_template(self, chat_history: list[dict], add_generation_prompt: bool = True) -> str:
        return HelixTokenizer.render_chat(chat_history, add_generation_prompt=add_generation_prompt)

    @property
    def tokenizer_name(self) -> str:
        return "corxlabs-helix"

    @torch.no_grad()
    def _logprobs(self, batch: list[tuple[list[int], list[int]]]) -> list[tuple[float, bool]]:
        """For each (context, continuation): the continuation's log-probability and whether it was greedy."""
        rows, lengths = [], []
        for context, continuation in batch:
            ids = (context + continuation)[-self.max_length :]
            rows.append(ids)
            lengths.append(len(ids))
        width = max(lengths)
        input_ids = torch.full((len(rows), width), self.tokenizer.pad_token_id, dtype=torch.long)
        attention = torch.zeros((len(rows), width), dtype=torch.long)
        for i, ids in enumerate(rows):
            input_ids[i, : len(ids)] = torch.tensor(ids)
            attention[i, : len(ids)] = 1
        logits = self.model(input_ids.to(self.device), attention_mask=attention.to(self.device)).logits.float()
        logprobs = torch.log_softmax(logits, dim=-1)
        out = []
        for i, (_, continuation) in enumerate(batch):
            n, total = len(continuation), lengths[i]
            targets = torch.tensor(rows[i][total - n :], device=self.device)
            positions = torch.arange(total - n - 1, total - 1, device=self.device)
            chosen = logprobs[i, positions, targets]
            greedy = logprobs[i, positions].argmax(-1).eq(targets).all()
            out.append((float(chosen.sum()), bool(greedy)))
        return out

    def loglikelihood(self, requests) -> list[tuple[float, bool]]:
        pairs = []
        for request in requests:
            context, continuation = request.args
            if context == "":
                context_ids, continuation_ids = [self.eot_token_id], self.tok_encode(continuation)
            else:
                whole, context_ids = self.tok_encode(context + continuation), self.tok_encode(context)
                continuation_ids = whole[len(context_ids) :] or self.tok_encode(continuation)
            pairs.append((context_ids, continuation_ids))
        results = []
        order = sorted(range(len(pairs)), key=lambda i: -(len(pairs[i][0]) + len(pairs[i][1])))
        scored = {}
        for start in range(0, len(order), self.batch_size):
            indices = order[start : start + self.batch_size]
            for index, result in zip(indices, self._logprobs([pairs[i] for i in indices]), strict=True):
                scored[index] = result
        for i in range(len(pairs)):
            results.append(scored[i])
        return results

    def loglikelihood_rolling(self, requests) -> list[float]:
        results = []
        for request in requests:
            (text,) = request.args
            ids = self.tok_encode(text)
            total = 0.0
            for start in range(0, len(ids), self.max_length - 1):
                window = ids[start : start + self.max_length - 1]
                context = [self.eot_token_id] if start == 0 else ids[start - 1 : start]
                total += self._logprobs([(context, window)])[0][0]
            results.append(total)
        return results

    @torch.no_grad()
    def generate_until(self, requests) -> list[str]:
        outputs = []
        for request in requests:
            context, options = request.args
            until = options.get("until", []) or []
            if isinstance(until, str):
                until = [until]
            max_new = int(options.get("max_gen_toks", 256))
            ids = torch.tensor([self.tok_encode(context)[-(self.max_length - max_new) :]], device=self.device)
            generated = self.model.generate(
                ids, max_new_tokens=max_new, temperature=0.0, eos_token_id=self.eot_token_id
            )
            text = self.tok_decode(generated[0, ids.shape[1] :].tolist())
            for stop in until:
                if stop and stop in text:
                    text = text[: text.index(stop)]
            outputs.append(text)
        return outputs


def run_lm_eval(
    model_dir: str, tasks: tuple[str, ...], limit: int | None = None, batch_size: int = 8, **kwargs
) -> dict:
    """``lm_eval.simple_evaluate`` on the tasks; returns ``{task: {metric: value}}`` in percent."""
    import lm_eval

    lm = HelixLM(model_dir, batch_size=batch_size, **kwargs)
    results = lm_eval.simple_evaluate(model=lm, tasks=list(tasks), limit=limit, log_samples=False)
    table = {}
    for task, metrics in results["results"].items():
        table[task] = {
            k.split(",")[0]: round(100 * v, 2)
            for k, v in metrics.items()
            if isinstance(v, (int, float)) and not k.endswith("stderr") and k != "alias"
        }
    return table


def run_evalplus(
    model_dir: str, dataset: str = "humaneval", limit: int | None = None, max_new_tokens: int = 768
) -> dict:
    """
    HumanEval+ / MBPP+: write a greedy completion for every problem, hand the file to EvalPlus, read back
    pass@1 on the base and the plus tests.
    """
    from evalplus.data import get_human_eval_plus, get_mbpp_plus, write_jsonl

    problems = get_human_eval_plus() if dataset == "humaneval" else get_mbpp_plus()
    lm = HelixLM(model_dir)
    samples = []
    for i, (task_id, problem) in enumerate(problems.items()):
        if limit is not None and i >= limit:
            break
        prompt = problem["prompt"]
        ids = torch.tensor([lm.tok_encode(prompt)], device=lm.device)
        generated = lm.model.generate(
            ids, max_new_tokens=max_new_tokens, temperature=0.0, eos_token_id=lm.eot_token_id
        )
        completion = lm.tok_decode(generated[0, ids.shape[1] :].tolist())
        for stop in ("\ndef ", "\nclass ", "\nif __name__", "\nprint(", "\n#"):
            if stop in completion:
                completion = completion[: completion.index(stop)]
        samples.append({"task_id": task_id, "solution": prompt + completion})
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / f"{dataset}_samples.jsonl"
        write_jsonl(str(path), samples)
        run = subprocess.run(
            [sys.executable, "-m", "evalplus.evaluate", "--dataset", dataset, "--samples", str(path)],
            capture_output=True, text=True, timeout=3600,
        )  # fmt: skip
    scores = {}
    for line in run.stdout.splitlines():
        lowered = line.lower()
        if "pass@1" in lowered:
            key = f"{dataset}_plus" if "plus" in lowered or "extra" in lowered else dataset
            with contextlib.suppress(ValueError, IndexError):
                scores[key] = round(100 * float(line.split("pass@1:")[-1].strip().split()[0]), 2)
    return scores or {"error": run.stdout[-500:] + run.stderr[-500:]}


def compare(ours: dict[str, float], name: str = "Helix") -> str:
    """One table: our scores beside the published targets, blank where a side has no number."""
    keys = sorted({k for scores in PUBLISHED.values() for k in scores} | set(ours))
    columns = [name, *PUBLISHED]
    lines = [f"{'benchmark':22}" + "".join(f"{c:>28}" for c in columns)]
    for key in keys:
        row = [ours.get(key)] + [PUBLISHED[c].get(key) for c in PUBLISHED]
        lines.append(f"{key:22}" + "".join(f"{(f'{v:.1f}' if v is not None else '-'):>28}" for v in row))
    lines.append("published figures are vendor-reported; re-run the targets with --baseline for a like-for-like table")
    return "\n".join(lines)


def evaluate_model(
    model_dir: str,
    tasks: tuple[str, ...] = DEFAULT_TASKS,
    evalplus: tuple[str, ...] = ("humaneval", "mbpp"),
    limit: int | None = None,
    out: str | None = None,
    log=print,
) -> dict:
    """Everything, then the comparison table; results also written to ``out`` as JSON."""
    started = time.time()
    scores: dict[str, float] = {}
    details: dict = {}
    if tasks:
        table = run_lm_eval(model_dir, tasks, limit=limit)
        details["lm_eval"] = table
        for task, metrics in table.items():
            for metric in ("acc_norm", "exact_match", "acc", "pass@1"):
                if metric in metrics:
                    scores[task] = metrics[metric]
                    break
    for dataset in evalplus:
        result = run_evalplus(model_dir, dataset, limit=limit)
        details[f"evalplus_{dataset}"] = result
        scores.update({k: v for k, v in result.items() if isinstance(v, (int, float))})
    text = compare(scores)
    log(text)
    record = {"model": model_dir, "scores": scores, "details": details, "seconds": time.time() - started}
    if out:
        Path(out).write_text(json.dumps(record, indent=1))
    return record
