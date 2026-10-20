# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
Command line entry point.

    helix info | demo | bench | recall          the architecture, on a laptop
    helix size                                  the ladder of named sizes and their parameter counts
    helix kernels                               which fused kernels this installation can use
    helix tokenizer train|sample                the CorxLabs tokenizer
    helix data verify|sample|prepare|status     the pretraining data
    helix train ...                             pretraining (torchrun for several GPUs)
    helix sft prepare|train ...                 the coding-agent fine-tune
    helix eval ...                              benchmarks and the comparison table
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import torch

from . import __author__, __version__
from .cache import HelixCache
from .config import HelixConfig, describe_sizes
from .model import HelixBraid, HelixForCausalLM, kernels_available

SMALL = {
    "vocab_size": 512,
    "hidden_size": 256,
    "intermediate_size": 512,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 64,
    "block_size": 64,
    "local_blocks": 3,
    "num_window_scales": 2,
    "index_layer_stride": 2,
    "index_topk": 8,
    "num_recurrent_heads": 2,
    "recurrent_head_dim": 64,
    "recurrent_value_head_dim": 64,
}


def _banner() -> str:
    return f"HELIX {__version__} — Hierarchical Episodic Linear IndeX — made by {__author__}"


def cmd_info(args: argparse.Namespace) -> None:
    config = HelixConfig(**SMALL)
    model = HelixForCausalLM(config)
    print(_banner())
    print("\nEvery block braids three sequence mixers over the same residual stream:\n")
    print(
        "  L  local      exact softmax attention over the last "
        f"{config.local_span} tokens, per-head windows {config.local_window_sizes}"
    )
    print(
        f"  R  recurrent  gated delta rule, {config.num_recurrent_heads} heads x "
        f"{config.recurrent_head_dim}x{config.recurrent_value_head_dim} matrix state"
    )
    print(
        f"  I  index      beam descent over a {config.index_branching}-ary landmark tree, "
        f"picks {config.index_topk} blocks of {config.block_size} tokens"
    )
    print(f"\nlayer schedule : {config.layer_types}")
    print(f"parameters     : {model.num_parameters():,}")
    keys_per_query = (config.local_blocks + 1 + config.index_topk) * config.block_size
    print(f"keys per query : {keys_per_query} on an indexed layer — constant, whatever the context length")
    print("\nRun `helix demo` to see it generate, `helix bench` for the scaling numbers, `helix size` for the ladder.")


def cmd_demo(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    config = HelixConfig(**{**SMALL, "vocab_size": args.vocab_size})
    model = HelixForCausalLM(config).eval()
    print(_banner())
    print(
        f"\nrandom-weight model, {model.num_parameters():,} parameters "
        f"(untrained, so the tokens below are noise — this shows the machinery runs)\n"
    )

    prompt = torch.randint(0, config.vocab_size, (1, args.prompt_length))
    started = time.perf_counter()
    generated = model.generate(prompt, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
    elapsed = time.perf_counter() - started
    print(f"prompt         : {prompt.shape[1]} tokens")
    print(f"generated      : {generated.shape[1] - prompt.shape[1]} tokens in {elapsed:.2f}s")
    print(f"first 16 ids   : {generated[0, prompt.shape[1] :][:16].tolist()}")

    # The property that makes HELIX worth using: the fixed-size decode state reproduces the full forward.
    reference = model(generated).logits
    cache = HelixCache(config)
    steps = [
        model(generated[:, i : i + 1], past_key_values=cache, use_cache=True).logits for i in range(generated.shape[1])
    ]
    delta = (torch.cat(steps, dim=1) - reference).abs().max().item()
    print(f"\nstep-by-step decoding vs one-shot forward: max difference {delta:.2e}")
    print(f"decode state after {generated.shape[1]} tokens: {cache.byte_size() / 1024:.1f} KiB")


def _count_pairs(model: HelixForCausalLM, length: int) -> int:
    total = 0
    for module in model.modules():
        if isinstance(module, HelixBraid):
            original = module._blocked_attention

            def counted(query_blocks, keys, *args, _original=original, **kwargs):
                nonlocal total
                batch, heads, blocks, block_size, _ = query_blocks.shape
                total += batch * heads * blocks * block_size * keys.shape[3]
                return _original(query_blocks, keys, *args, **kwargs)

            module._blocked_attention = counted
    with torch.no_grad():
        model(torch.zeros(1, length, dtype=torch.long))
    return total


def _slope(xs: list[int], ys: list[float]) -> float:
    lx, ly = [math.log(v) for v in xs], [math.log(v) for v in ys]
    mx, my = sum(lx) / len(lx), sum(ly) / len(ly)
    return sum((a - mx) * (b - my) for a, b in zip(lx, ly, strict=True)) / sum((a - mx) ** 2 for a in lx)


def cmd_bench(args: argparse.Namespace) -> None:
    torch.manual_seed(0)
    # The reference path counts attention pairs exactly; the fused path does not expose them.
    model = HelixForCausalLM(HelixConfig(**SMALL, attention_backend="sdpa")).eval()
    print(_banner())
    print("\nAttention pairs are exact: the query/key products the forward pass actually forms, counted by")
    print("instrumenting the attention call. Flat per-token means linear in the context length.\n")
    print(f"{'tokens':>8}  {'pairs':>16}  {'per token':>10}  {'seconds':>8}  {'cache MiB':>10}")

    lengths, times = args.lengths, []
    for length in lengths:
        pairs = _count_pairs(model, length)
        ids = torch.zeros(1, length, dtype=torch.long)
        with torch.no_grad():
            model(ids)
            started = time.perf_counter()
            cache = model(ids, use_cache=True).past_key_values
            elapsed = time.perf_counter() - started
        times.append(elapsed)
        print(
            f"{length:>8}  {pairs:>16,}  {pairs / length:>10,.0f}  {elapsed:>8.2f}  "
            f"{cache.byte_size() / 1024**2:>10.1f}",
            flush=True,
        )

    if len(lengths) > 1:
        print(
            f"\nwall clock fits N^{_slope(lengths, times):.2f}   "
            "(full attention is N^2.00 in pairs; this is a reference implementation with no fused kernel)"
        )


def cmd_recall(args: argparse.Namespace) -> None:
    """Train two tiny models on associative recall: with the index strand, and without it."""
    from .recall import run_recall

    print(_banner())
    run_recall(steps=args.steps, seq_len=args.seq_len, pairs=args.pairs, batch_size=args.batch_size, seed=args.seed)


def cmd_size(args: argparse.Namespace) -> None:
    print(_banner())
    print(
        "\nThe ladder. Parameters from the shapes (65,536-token vocabulary); 'tokens x20' is the Chinchilla budget.\n"
    )
    print(describe_sizes())
    if args.size:
        config = HelixConfig.named(args.size)
        flops = config.flops_per_token()
        print(f"\n{args.size}: {config.num_parameters() / 1e9:.2f}B parameters, {flops:.3e} training FLOPs a token")
        for gpus, peak, name in ((8, 2.25e15, "B300"), (8, 9.89e14, "H100"), (1, 2.25e15, "B300")):
            for mfu in (0.35, 0.45):
                tokens = 20 * config.num_parameters()
                hours = tokens * flops / (gpus * peak * mfu) / 3600
                print(
                    f"  {gpus} x {name} at {mfu:.0%} MFU: {tokens / 1e9:.0f}B tokens in {hours:,.0f} h "
                    f"({hours / 24:.1f} days)"
                )


def cmd_kernels(args: argparse.Namespace) -> None:
    print(_banner())
    for name, available in kernels_available().items():
        print(f"  {name:24} {'available' if available else 'not installed'}")
    if torch.cuda.is_available():
        print(f"  cuda                     {torch.cuda.device_count()} x {torch.cuda.get_device_name(0)}")
    else:
        print("  cuda                     no GPU visible")


def cmd_tokenizer(args: argparse.Namespace) -> None:
    from .data.prepare import sample_text
    from .tokenizer import iter_text_sample, train_tokenizer

    if args.action == "sample":
        out = sample_text(args.data, bytes_per_source=int(args.bytes_per_source), token=os.environ.get("HF_TOKEN"))
        print(f"sample written to {out}")
        return
    files = [Path(p) for p in args.input] or [Path(args.data) / "tokenizer_sample.txt"]
    out_dir = Path(args.out or (Path(args.data) / "tokenizer"))
    print(f"training a {args.vocab_size:,}-token tokenizer on {[str(f) for f in files]} -> {out_dir}")
    started = time.time()
    tokenizer = train_tokenizer(
        (line.replace("\\n", "\n") for line in iter_text_sample(files, int(args.max_bytes))),
        out_dir, vocab_size=args.vocab_size, show_progress=True,
    )  # fmt: skip
    print(f"done in {time.time() - started:.0f}s: {tokenizer.vocab_size:,} tokens")


def cmd_data(args: argparse.Namespace) -> None:
    from .data.catalog import PRETRAIN_SOURCES, SFT_SOURCES, describe_report, verify_sources
    from .data.prepare import describe_prepared, prepare_all

    token = os.environ.get("HF_TOKEN")
    if args.action == "verify":
        print(describe_report(verify_sources(PRETRAIN_SOURCES, token)))
        print()
        print(describe_report(verify_sources(SFT_SOURCES, token)))
    elif args.action == "prepare":
        tokenizer = args.tokenizer or str(Path(args.data) / "tokenizer")
        prepared = prepare_all(
            args.data,
            tokenizer,
            int(args.tokens),
            workers=args.workers,
            token=token,
            only=tuple(args.only),
            min_free_gb=args.min_free_gb,
        )
        print(f"[data] prepared: {prepared}")
        print(describe_prepared(args.data))
        print("[data] PREPARED", flush=True)
    elif args.action == "status":
        print(describe_prepared(args.data))


def cmd_train(args: argparse.Namespace) -> None:
    from .train import main as train_main

    raise SystemExit(train_main(args.train_args))


def cmd_sft(args: argparse.Namespace) -> None:
    if args.action == "prepare":
        from .data.agent import prepare_sft

        tokenizer = args.tokenizer or str(Path(args.data) / "tokenizer")
        out = prepare_sft(args.data, tokenizer, seq_len=args.seq_len, max_conversations=args.max_conversations,
                          token=os.environ.get("HF_TOKEN"))  # fmt: skip
        print(f"[sft] PREPARED {out}", flush=True)
    else:
        from .train import main as train_main

        sft_dir = str(Path(args.data) / "sft")
        extra = ["--data", sft_dir, "--mixture", str(Path(sft_dir) / "mixture.json"), "--tokenizer",
                 str(Path(sft_dir) / "tokenizer"), *args.train_args]  # fmt: skip
        raise SystemExit(train_main(extra))


def cmd_eval(args: argparse.Namespace) -> None:
    from .evaluate import evaluate_model

    evaluate_model(
        args.model, tasks=tuple(args.tasks), evalplus=tuple(args.evalplus), limit=args.limit, out=args.out,
        safety=args.safety,
    )  # fmt: skip


def cmd_jamaica(args: argparse.Namespace) -> None:
    if args.action == "crawl":
        from .data.jamaica_gov import crawl

        crawl(args.data, hours=args.hours, max_pages_per_host=args.max_pages, delay=args.delay)
        print("[crawl] CRAWLED", flush=True)
    else:
        from .data.synth import Teacher, synthesize

        teacher = Teacher(args.teacher_url, args.model, api_key=os.environ.get("TEACHER_API_KEY", "none"))
        synthesize(args.data, teacher, count=args.count, workers=args.workers, seed=args.seed)
        print("[synth] SYNTHESIZED", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="helix", description=_banner())
    parser.add_argument("--version", action="version", version=f"helix-lm {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("info", help="what the architecture is and what it costs").set_defaults(func=cmd_info)

    demo = sub.add_parser("demo", help="build an untrained model, generate, verify the decode state")
    demo.add_argument("--prompt-length", type=int, default=64)
    demo.add_argument("--max-new-tokens", type=int, default=32)
    demo.add_argument("--temperature", type=float, default=0.0)
    demo.add_argument("--vocab-size", type=int, default=512)
    demo.add_argument("--seed", type=int, default=0)
    demo.set_defaults(func=cmd_demo)

    bench = sub.add_parser("bench", help="cost against context length")
    bench.add_argument("--lengths", type=int, nargs="+", default=[1024, 2048, 4096, 8192])
    bench.set_defaults(func=cmd_bench)

    recall = sub.add_parser("recall", help="train on associative recall, with and without the index strand")
    recall.add_argument("--steps", type=int, default=600)
    recall.add_argument("--seq-len", type=int, default=256)
    recall.add_argument("--pairs", type=int, default=16)
    recall.add_argument("--batch-size", type=int, default=16)
    recall.add_argument("--seed", type=int, default=0)
    recall.set_defaults(func=cmd_recall)

    size = sub.add_parser("size", help="the named sizes and what they cost to train")
    size.add_argument("size", nargs="?", default="")
    size.set_defaults(func=cmd_size)

    sub.add_parser("kernels", help="which fused kernels are available here").set_defaults(func=cmd_kernels)

    tokenizer = sub.add_parser("tokenizer", help="the CorxLabs tokenizer")
    tokenizer.add_argument("action", choices=["sample", "train"])
    tokenizer.add_argument("--data", required=True, help="data directory")
    tokenizer.add_argument("--input", nargs="*", default=[], help="text files (default: <data>/tokenizer_sample.txt)")
    tokenizer.add_argument("--out", default="", help="output directory (default: <data>/tokenizer)")
    tokenizer.add_argument("--vocab-size", type=int, default=65536)
    tokenizer.add_argument("--max-bytes", type=float, default=20e9, help="at most this much text is read")
    tokenizer.add_argument("--bytes-per-source", type=float, default=400e6, help="for `sample`")
    tokenizer.set_defaults(func=cmd_tokenizer)

    data = sub.add_parser("data", help="verify, prepare and inspect the pretraining data")
    data.add_argument("action", choices=["verify", "prepare", "status"])
    data.add_argument("--data", default="", help="data directory")
    data.add_argument("--tokenizer", default="")
    data.add_argument("--tokens", type=float, default=460e9, help="tokens to prepare in total (the run's budget)")
    data.add_argument("--workers", type=int, default=max(4, (os.cpu_count() or 8) - 4))
    data.add_argument("--only", nargs="*", default=[], help="only these sources")
    data.add_argument(
        "--min-free-gb",
        type=float,
        default=300.0,
        help="stop preparing when the volume has less free space than this (room for checkpoints)",
    )
    data.set_defaults(func=cmd_data)

    train = sub.add_parser("train", help="pretrain (see `helix train --help`)")
    train.add_argument("train_args", nargs=argparse.REMAINDER)
    train.set_defaults(func=cmd_train)

    sft = sub.add_parser("sft", help="prepare the coding-agent data, or fine-tune on it")
    sft.add_argument("action", choices=["prepare", "train"])
    sft.add_argument("--data", required=True, help="data directory (the SFT set lives under <data>/sft)")
    sft.add_argument("--tokenizer", default="")
    sft.add_argument("--seq-len", type=int, default=32768)
    sft.add_argument("--max-conversations", type=int, default=300_000, help="per source")
    sft.add_argument("train_args", nargs=argparse.REMAINDER, help="for `train`: passed to `helix train`")
    sft.set_defaults(func=cmd_sft)

    evaluate = sub.add_parser("eval", help="benchmarks and the comparison table")
    evaluate.add_argument("--model", required=True, help="a saved model directory (an export)")
    evaluate.add_argument(
        "--tasks", nargs="*", default=["hellaswag", "arc_challenge", "gsm8k", "mmlu_pro", "humaneval"]
    )
    evaluate.add_argument("--evalplus", nargs="*", default=["humaneval", "mbpp"])
    evaluate.add_argument("--limit", type=int, default=None, help="problems per task (a quick look)")
    evaluate.add_argument("--out", default="", help="write the results here as JSON")
    evaluate.add_argument(
        "--no-safety", dest="safety", action="store_false", help="skip the refusal / over-refusal check"
    )
    evaluate.set_defaults(func=cmd_eval)

    jamaica = sub.add_parser(
        "jamaica", help="the Jamaican data: crawl public records, synthesize code-switched dialogues"
    )
    jamaica.add_argument("action", choices=["crawl", "synth"])
    jamaica.add_argument("--data", required=True)
    jamaica.add_argument("--hours", type=float, default=6.0, help="crawl: wall-clock cap")
    jamaica.add_argument("--max-pages", type=int, default=20_000, help="crawl: per host")
    jamaica.add_argument("--delay", type=float, default=2.0, help="crawl: seconds between requests to one host")
    jamaica.add_argument("--count", type=int, default=100_000, help="synth: dialogues to make")
    jamaica.add_argument("--teacher-url", default="http://127.0.0.1:8000/v1", help="synth: OpenAI-compatible endpoint")
    jamaica.add_argument("--model", default="Qwen/Qwen3-32B", help="synth: the teacher model name at the endpoint")
    jamaica.add_argument("--workers", type=int, default=32)
    jamaica.add_argument("--seed", type=int, default=0)
    jamaica.set_defaults(func=cmd_jamaica)

    args = parser.parse_args(argv)
    if getattr(args, "train_args", None) and args.train_args[:1] == ["--"]:
        args.train_args = args.train_args[1:]
    try:
        args.func(args)
    except BrokenPipeError:
        # `helix info | head` closes the pipe early; exit quietly instead of dumping a traceback.
        try:
            sys.stdout.close()
        finally:
            os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return 0
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
