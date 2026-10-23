# HELIX

**H**ierarchical **E**pisodic **L**inear **I**nde**X** — a sequence architecture that keeps what attention
is good at while dropping the quadratic bill.

```bash
pip install helix-lm
```

```python
import torch
from helix_lm import HelixConfig, HelixForCausalLM

model = HelixForCausalLM(HelixConfig(vocab_size=32000, hidden_size=1024, num_hidden_layers=12))
logits = model(torch.randint(0, 32000, (1, 8192))).logits
text = model.generate(torch.randint(0, 32000, (1, 64)), max_new_tokens=32, temperature=0.8)
```

```bash
helix info      # what it is and what it costs
helix demo      # build a model, generate, verify the decode state
helix bench     # cost against context length
helix recall    # train with and without the index strand, and compare
```

---

## The idea

A transformer compares every token to every other token. Ten thousand tokens is fifty million
comparisons; double the input and you quadruple the work. Linear-time alternatives fix that by squeezing
the past into a fixed-size state — and then they cannot quote you a name they saw once, because a state of
`S` bits cannot tell apart more than `2^S` histories. That is a counting argument, not an engineering gap.

HELIX does not try to compress its way around it. It **separates memory capacity from memory bandwidth**.
Every block braids three mixers over the same residual stream:

| strand | reads | training cost | held at decode |
| --- | --- | --- | --- |
| **L** local | the last `local_blocks × block_size` tokens, exactly, with a different window per head | `O(N · span)` | a window-sized ring buffer |
| **R** recurrent | everything, compressed into a matrix-valued delta-rule state | `O(N · d_k · d_v)` | one matrix + two conv states per head |
| **I** index | `index_topk` blocks chosen anywhere in the past by a beam descent over a landmark tree | `O(N · topk · block_size)` + `O(N log N)` routing | the key/value history and a landmark tree, of which a step reads `O(topk · block_size + branching · log N)` |

Storage is `O(N)`, append-only, never rewritten, and can live in cold memory. What a single token *touches*
is bounded. So the claim is **`O(1)` hot state and `O(log N)` probes** — not `O(1)` total memory, which is
impossible for anything that can quote an exact name from a million tokens back.

## How the index works

The past is cut into blocks of `block_size` tokens. Each closed block gets a **landmark** — a learned pooled
summary that mixes a plain mean with an attention pool, because a mean alone erases the rare token a later
query will be hunting for. Landmarks are pooled again, `index_branching` at a time, into a tree, using one
weight-shared pooler at every internal level.

Two rules keep it causal *and* parallel:

1. **A node is eligible for query block `J` only if the whole span it summarizes ends at or before `J`.** A
   summary is never consulted by a query it partly describes, so a landmark is computed once and reused by
   every query.
2. **Routing for block `J` comes from the hidden state at the end of block `J − 1`.** Strictly past for
   every token of block `J`, so one gather serves `block_size` queries. Anything routing could not
   anticipate inside that block is covered exactly by strand L.

The descent keeps `index_beam_width` nodes per level. Candidates at each level are the beam's children
**plus** that level's *frontier* — eligible nodes whose parent is not eligible. At most `branching − 1` per
level, and the frontiers across all levels tile `[0, J)` exactly, so nothing reachable is cut off by a beam
that descended elsewhere. Selection scores are fed back as additive attention logits, accumulated along the
descent path, which is what lets gradient reach every level of the tree rather than just the leaves.

## What has been measured

On small models, 4 CPU cores. Reproduce with `helix bench` and `helix recall`.

**Cost against context length.** Attention pairs are exact — the query/key products the forward pass
actually forms, counted by instrumenting the attention call.

| tokens | HELIX pairs/token | full attention pairs/token | HELIX | full attention |
| ---: | ---: | ---: | ---: | ---: |
| 4 096 | 8 192 | 32 776 | 0.4 s | 0.2 s |
| 8 192 | 8 192 | 65 544 | 0.9 s | 0.6 s |
| 16 384 | 8 192 | 131 080 | 1.7 s | 1.9 s |
| 32 768 | 8 192 | 262 152 | 4.3 s | 6.7 s |

HELIX fits `N^1.00` in attention pairs; full attention fits `N^2.00`. By 32k tokens HELIX forms **32×
fewer** and runs **1.6× faster** — as unfused reference PyTorch against a fused SDPA kernel.

**Recall from far back.** Multi-query associative recall: bindings written near the start, queried ~200
tokens later, well outside the local window.

| model | recall | 95% CI |
| --- | ---: | :---: |
| HELIX (L+R+I) | **30.8%** | [28.5%, 33.1%] |
| HELIX no index (L+R) | 6.0% | [4.8%, 7.2%] |
| full attention | 32.0% | [29.7%, 34.3%] |
| *chance* | *6.2%* | |

Read the two gaps separately. **HELIX against its own ablation is the result**: strip the index strand and
the same model, same width, same depth, sits at chance — its window cannot see the bindings and its
recurrent state cannot hold them. That gap is ~21 standard errors. **HELIX against full attention is a
tie**, not a win or a loss: 1.2 points with a 1.7-point standard error, z = 0.72.

## Correctness

`pytest` covers the properties you can check without training anything:

- **strictly causal** — editing token `t` leaves every earlier logit bitwise unchanged;
- **decode == prefill** — token-by-token generation from the fixed-size state reproduces the one-shot
  forward, as does unaligned chunked prefill;
- **right padding is exactly invariant**;
- **constant work per token** — each query reads the same number of keys at any context length;
- **the cache is what the architecture claims** — fixed-size recurrent and convolution states everywhere,
  a window-capped key/value cache on non-indexing layers;
- **tiling is bitwise inert** — `attention_tile_blocks` caps peak activation memory and changes nothing;
- **every parameter gets gradient**, landmark poolers included.

## What this is not

**There are no trained checkpoints.** Nobody has trained HELIX on real text. Everything above is about
complexity, causality and recall mechanics — properties provable on an untrained model or a synthetic
probe — not about perplexity or language quality. The inductive-bias arguments (multi-scale windows, short
convolutions, surprise-gated writes) are untested hypotheses.

Treat this as an architecture proposal with a working, tested reference implementation, not a better model.
Finding out whether it is actually better needs real training runs on real hardware.

Known design limits, stated plainly:

- One retrieval serves a whole query block, so `index_topk` must cover the diversity a `block_size` span
  asks for.
- Routing is one block stale — the index cannot react to the token currently being generated.
- Landmarks are pooled from keys that already carry their rotary phase, so routing is not purely
  content-addressed.
- Left padding shifts the block grid, so it is not bit-identical to an unpadded run. Right padding is.
- No fused kernel for the gathered attention yet; the scaling is right, the constant factor is not.

## Configuration

Every knob is on `HelixConfig`, documented in its docstring. The ones that matter most:

```python
HelixConfig(
    block_size=64,          # memory-block granularity
    local_blocks=4,         # strand L sees 4 previous blocks plus its own
    num_window_scales=4,    # head groups get geometrically different windows
    index_topk=8,           # blocks strand I retrieves per query block
    index_layer_stride=3,   # ...on every third layer; the rest keep an O(1) KV cache
    index_branching=8,      # landmark tree fan-out
    attention_tile_blocks=64,  # caps peak activation memory; no effect on the result
)
```

## 2.0: ANANSI, faster learning, audio

2.0 is a major upgrade documented in [`docs/ANANSI.md`](docs/ANANSI.md), with the evidence behind every piece:

* **faster learning** in the trainer: `--optimizer muon` (orthogonalized momentum, about 2x AdamW's token
  efficiency in Moonshot's scans, Kimi K2's optimizer), `--mup` (learning-rate transfer from a pilot),
  multi-token prediction, `helix grow` (train small, stack, continue), `--reweight-strength` (online data-mixture
  reweighting from held-out progress);
* **ANANSI** (`helix_lm.anansi`, sizes `a-350m` .. `a-34b`): the HELIX strands in a new body -- four residual
  streams mixed by manifold-constrained hyper-connections, a looped reasoning core with adaptive depth at
  inference (`model.think`), gated attention, a hashed n-gram memory, sandwich norms, a recurrence-only prelude,
  an optional fine-grained mixture of experts, and a multi-token predictor;
* **audio** (`helix_lm.audio`): a raw-waveform codec at 25 frames a second with a split residual quantizer, a
  speaker encoder for zero-shot voice cloning, score streams for singing, and an omni wrapper that interleaves
  text tokens and audio frames in one Anansi sequence;
* **a toy task suite** (`helix anansi bench`): recall, parity, multi-hop, copy-with-noise for the three families.

```bash
helix anansi sizes && helix anansi demo
helix anansi bench --steps 300
helix train --size a-1b --optimizer muon --mup --reweight-strength 1 --data /data/corx --out /data/corx/runs/a-1b
helix audio sources --commercial
```

## 2.2: hardening TriStream-2

2.2 answers the three failure modes named against 2.1 and the robotic-sound problem, with mechanisms, detectors
and fallbacks (`docs/TRISTREAM.md` section 12):

* **the refiner** (`helix_lm.audio.refiner`): phase-true additive harmonics from a 100 Hz F0, 10 ms shaped noise
  for consonants and breath, a snake-activated residual network and a gate on the codec's output, trained with
  transient and high-band losses; a 50 Hz codec option;
* **stream sync**: sub-frame F0 in the source stream, a monotonic alignment prior on lyric attention, a formant /
  pitch invariance loss;
* **routing**: look-ahead conditioning and a song-event track so the index is one block early instead of one
  block stale; sway sampling, adaptive steps, classifier-free guidance;
* **the ear** (`helix_lm.music.artifacts`): frame-rate buzz, harmonic coherence, metallic high band, consonant
  smear, sync lag, robotic pitch;
* **the humanizer** (`helix_lm.music.humanize`): portamento, overshoot, scoop, late and drifting vibrato, wander,
  jitter, releases, attacks, phrase arches, breaths; styles; the conductor renders through it;
* **the fail-safe loop** (`helix_lm.music.failsafe`): generate, listen, escalate, repair the span, transplant the
  source's unvoiced band, fall back to the source-filter sketch, report;
* a per-band **room model** so a swapped voice gets the original's room, not a broadband tail;
* **2.3**: the molab program mirrors its run to a private Hub repo and resumes from it (checkpoints with
  step and hours, budgets that carry across restarts, retries), fills the GPU (auto batch, TF32, prefetch),
  extracts and prepares the downloaded sets on the GPU, and keeps speech from swamping singing: domain
  shares first (speech annealed 50 % -> 10 % of each batch), spoken clips at half loss weight, a 50/50 codec,
  singing-only evaluation; round three (section 14): a sustain-aware alignment prior for melismas, a codebook
  register audit with high-register oversampling, rooms and EQ on every training input against dry targets,
  a per-frame formant curve with a pop detector, and a two-block look-ahead switched on before drops;
* **2.3.1**: downloads verified by size and md5 with resume and a second route, corrupt archives deleted
  and fetched again, one run per root (`run.lock`), the data stage retried until every source is whole;
* **round two** (`docs/TRISTREAM.md` section 13): one global frame grid and measured drift control for long
  tracks (`music.longform`), register physics with a bounded formant move and F1 tuning (`music.register`), an
  air-band extension to 48 kHz with air and pre-echo detectors, an intake that measures and cleans real-world
  audio and passes its quality to the model as a condition (`music.intake`), and robust reference embeddings
  that drop shouted, whispered or buried windows and report contamination.

## 2.1: TriStream-2 and the music stack

2.1 adds `helix_lm.music`, documented with its evidence in [`docs/TRISTREAM.md`](docs/TRISTREAM.md):

* **TriStream-2** (`ts-765m`, 764.5M parameters): a source / filter / noise / lyric singing model on the Anansi
  trunk with a rectified-flow decoder over codec frames and a source-filter sketch as its prior; in-context
  zero-shot cloning through masked-span editing; singer **and accent** adversaries; pitch and duration heads;
* **the song analyser**: beats and downbeats, key and scale, chords and tension, structure, loudness,
  brightness, room, groove and vibe cues, in plain PyTorch;
* **the frisson critic**: a per-beat "chills" score from the triggers the literature found (dynamic surges, a
  voice entering, range expansion, surprise inside a predictable context, tension released, the dip before the
  drop, groove), with the weights written out as priors and a head to learn them from labels;
* **the conductor**: a readable vocal plan (where to enter, which notes in key and in range, harmonies, breath /
  power / vibrato / warmth / echo sliders, the sudden soloist, the octave-leap climax) scored for restraint;
* **separation, conversion, effects**: a band-split separator with mask budgeting, harmonic gating, target-singer
  extraction for duets and a bleed report; the voice-swap pipeline (separate, analyse, capture the vocal's
  effects, convert, re-apply, remix, gates); reverb / echo / muffle / retune / formant shift / ducking / limiter /
  breaths; glitch detection and repair;
* **data rules**: accent tags and balancing, pitch-range buckets with a gaps report, dry-target checks,
  singer-disjoint splits, augmentation into the hard input conditions, multi-file uploads;
* **the 96 GB program** (`helix tristream molab`, `python -m helix_lm.scripts.tristream_molab`): data, prepare,
  codec, refiner, air, train, eval, push, swap on one RTX PRO 6000 Blackwell, resumable, with auto batch sizing
  and the fail-safe loop in eval and swap.

```bash
pip install "helix-lm[music]"
helix tristream sizes && helix tristream demo
helix music analyze song.wav && helix music plan song.wav --out plan.json
helix music swap song.wav reference.wav --out out.wav
python -m helix_lm.scripts.tristream_molab --root /workspace/tristream --stage all --hf-repo CorxLabs/Tristream-V2-SVS
```

Nothing in 2.1 has been trained at size; the document says what is measured (toy scale) and what is a plan.

## Training Helix (1.1): CorX5.5

1.0 added the engine that turns the architecture into a model and the ladder of named sizes; 1.1 is the
program that trains **CorX5.5**, CorxLabs' 34B long-horizon coding agent, from scratch on 870B tokens in one
year on 8 x B300, largely on Jamaican data, with safeguards. The plan, arithmetic, data mixture, safeguards
and limits are in [`docs/CORX5.5.md`](docs/CORX5.5.md); the earlier 23B plan is kept in
[`docs/HELIX_23B_PLAN.md`](docs/HELIX_23B_PLAN.md). The whole program is one startup script
(`corx_startup.sh`, built from the wheel) that runs as a service and pushes code, checkpoints and milestones
to `CorxLabs/CorX5.5`.

```bash
pip install "helix-lm[all]"
helix size 34b                  # the ladder: 350m, 1b, 3b, 7b, 23b, 34b -- parameters, FLOPs, hours on 8 x B300
helix kernels                   # FlexAttention / flash-linear-attention availability
helix jamaica crawl --data /data/corx --hours 12          # Jamaican public records (polite, robots.txt)
helix jamaica synth --data /data/corx --count 300000      # SJE/Patois code-switched dialogues via a local vLLM teacher
```

| piece | module | what it does |
| --- | --- | --- |
| fused strand L | `model.py` (`attention_backend="flex"`) | each head's sliding window as one block-sparse FlexAttention kernel, no key copies; equal to the reference staircase |
| Triton strand R | `model.py` (`recurrent_backend="fla"`) | `flash-linear-attention`'s chunked gated delta rule, the reference as fallback |
| sizes | `config.py` (`HelixConfig.named`) | 350m / 1b / 3b / 7b / 23b / 34b with analytic parameter and FLOP counts |
| Anansi | `anansi/` | the 2.0 network: hyper-connected streams, looped core, gated attention, n-gram memory, MoE option, MTP |
| optimizers | `optim/` | Muon + AdamW split, muP plans |
| growth | `growth.py` | identity and stack growth of exports |
| audio | `audio/` | codec, speaker encoder, singing streams, omni wrapper, dataset catalogue |
| tasks | `tasks.py` | the toy suite |
| transformer twins | `arch.py` | `t-350m` .. `t-34b` on the Qwen3 layout (QK-norm), same trainer, exported as `Qwen3ForCausalLM`; the pilot's control |
| tokenizer | `tokenizer.py` | the CorxLabs 65,536-token byte-level BPE (uint16 storage), chat and tool-call format |
| data | `data/` | verified Hub catalog (36 pretraining sources: web, code, execution traces, commits and issues, math, legal, health, finance, education, Patois), row renderers, streaming prepare to shards with a free-space guard, the Jamaica and Caribbean filters, identity documents, a deterministic resumable mixture loader |
| Jamaican data | `data/jamaica_gov.py`, `data/synth.py` | a polite crawler of Jamaican public records; SJE/Patois code-switched dialogues for banking, legal, health, education and customer service from a local Apache-2.0 teacher |
| safeguards | `data/safety.py` | the policy prompt, boundary and look-alike examples, agent conduct, CoCoNot/tulu refusals, XSTest over-refusal and harmful-set refusal rates in `helix eval` |
| trainer | `train.py`, `hub.py` | FSDP2 bf16, fused AdamW, warmup-stable-decay fitted to `--hours`, sharded checkpoints, bf16 safetensors export pushed to the Hub (history squashed, milestone copies every N tokens), resume, MFU, a stop file |
| SFT | `data/agent.py` | SWE-agent and SWE-smith trajectories, OpenCodeInstruct, OpenThoughts, smoltalk, tulu into one masked chat format |
| eval | `evaluate.py` | lm-evaluation-harness adapter, EvalPlus, a table against the published Qwen numbers |

```bash
# on the training machine (8 GPUs), in this order; every step resumes where it stopped
helix data verify                                     # sources reachable with $HF_TOKEN
helix tokenizer sample --data /data/helix             # a text sample of the mixture ...
helix tokenizer train  --data /data/helix             # ... becomes the 65,536-token tokenizer
helix data prepare --data /data/helix --tokens 23e9   # shards for a 1b run (23B tokens = 20 x 1.17B)
torchrun --nproc_per_node=8 -m helix_lm.train --size 1b --data /data/helix --out /data/helix/runs/helix-1b \
    --hours 7 --hub-repo CorxLabs/Helix-1B
helix sft prepare --data /data/helix                  # the coding-agent conversations, masked and packed
helix sft train --data /data/helix -- --size 1b --out /data/helix/runs/helix-1b-sft \
    --init-from /data/helix/runs/helix-1b/export --seq-len 32768 --hours 2
helix eval --model /data/helix/runs/helix-1b-sft/export
```

What the tests check (`pytest`, on a CPU, in about three minutes): the fused local attention equals the
reference; the chunked loss equals the one-piece loss; recomputation changes nothing; the sharded
safetensors round-trip; the analytic parameter count matches the built model; the tokenizer round-trips,
fits in uint16 and masks a chat on the assistant's turns only; files become shards, Jamaica is found,
the identity is written; the loader is deterministic, striped across ranks, resumable and capped per
source; a training run checkpoints, exports and **resumes to the same losses and weights**; two
processes train with FSDP2 and distributed checkpoints and resume; SFT data is masked through packing and
a fine-tune starts from a saved model; the lm-eval adapter's log-likelihoods match a direct forward.

**Still true: there are no trained checkpoints yet.** 1.1 is the engine and the program, tested at toy scale
on a CPU. The program's first rung on the real machine is a 25-minute smoke run and its second a 1B pilot of
HELIX against its transformer twin on the same tokens; whether HELIX trains as well as a transformer on text
is what that pilot measures, before the year-long run starts.

## License

Apache-2.0. Portions of the delta-rule and rotary helpers derive from
[HuggingFace Transformers](https://github.com/huggingface/transformers) (Apache-2.0); see `NOTICE`.

**Made by Nathan.**
