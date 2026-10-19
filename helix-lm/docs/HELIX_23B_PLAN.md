# Helix-23B: the plan

CorxLabs' coding agent: a 23-billion-parameter HELIX model, pretrained from scratch on 467 billion tokens
(20 tokens a parameter -- the Chinchilla budget), fine-tuned on coding-agent trajectories, built and trained on
one 8 x B300 machine, with Jamaica in its data and its identity. This document is the arithmetic behind it,
the ladder of runs that leads to it, the data, and what "competing with Qwen3.5-27B" can and cannot mean.
Every number here comes from `helix size 23b` and the catalog in `helix_lm/data/catalog.py`.

## 1. The model

`HelixConfig.named("23b")`: 48 blocks, hidden 6,144, 48 query heads of 128 over 8 key/value groups, SwiGLU of
15,360, 24 recurrent heads with a 128 x 128 matrix state each, 65,536-token vocabulary.
**23.34B parameters, 22.53B of them outside the embeddings.** Each block braids the three HELIX strands:

| strand | what it reads on every token | cost per token |
| --- | --- | --- |
| L, local | an exact window of up to 1,024 tokens, on four scales across the heads (128 / 256 / 512 / 1,024) | 1,024 keys at most |
| R, recurrent | everything, through a gated delta-rule matrix state (24 x 128 x 128 numbers) | fixed |
| I, index (every other block) | 16 blocks of 128 tokens chosen anywhere in the past by a landmark tree | 2,048 keys |

So a token on an indexed layer touches about 3,000 keys whatever the context length. A full-attention model
touches 8,192 at an 8k context and 131,072 at 128k. That is the reason to build an agent on HELIX: long
sessions cost what short ones cost, and the decode state does not grow.

Training FLOPs per token: **1.434e11** (`6N` over the non-embedding weights is 1.352e11; the window, index and
recurrence terms add 6%).

## 2. What it costs

8 x B300 at the vendor's 2.25 PFLOPS dense BF16 each is 18 PFLOPS. Model-FLOP utilization (MFU) of 35-45% is what
well-tuned dense training reaches; this engine has no fused kernel for the index strand yet, so plan on the low end
until the pilot measures it. The hourly rate below is 8 x $5.87, Spheron's single-B300 price; read the real one off the
dashboard.

| size | parameters | tokens (20x) | FLOPs | hours at 35% MFU | days | ~cost at $47/h |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 350m | 0.36B | 7B | 1.5e19 | 0.7 | 0.03 | $31 |
| 1b | 1.17B | 23B | 1.6e20 | 7.0 | 0.3 | $330 |
| 3b | 2.74B | 55B | 9.3e20 | 41 | 1.7 | $1,900 |
| 7b | 7.63B | 153B | 7.1e21 | 314 | 13 | $14,800 |
| **23b** | **23.34B** | **467B** | **6.7e22** | **2,950** | **123** | **$139,000** |

At 45% MFU the 23b run is 96 days and about $108,000. Data preparation adds about a day of the machine's CPUs
(it runs beside the first pilots). **The 23B run is a four-month, six-figure project on this machine.** Nothing
in the engine makes that smaller; only money, more machines, or a smaller model does.

What a given budget buys, if the size is chosen to match it (compute-optimal, 35% MFU):

| hours on 8 x B300 | cost | best model for that compute |
| ---: | ---: | --- |
| 24 | $1,100 | ~2B parameters on ~43B tokens |
| 48 | $2,300 | ~3B on ~60B |
| 120 | $5,600 | ~5B on ~95B |
| 240 | $11,000 | ~7B on ~135B |

## 3. The ladder

Nobody has trained HELIX on real text before this. The ladder exists so that the first time the architecture meets
language is a $30 run, not a $139,000 one, and so that each rung either earns the next or stops the program.

1. **Smoke, 20 minutes.** `helix-350m` on the first prepared shards. Checks: loss falls, FSDP2 and both kernels
   run on the B300s (FlexAttention; flash-linear-attention if it compiles on sm_103, the reference otherwise), the
   checkpoint saves, exports and pushes, a resume continues the same curve. Nothing else is started until this passes.
2. **Pilot, 7 hours, `helix-1b` on 23B tokens.** The first real number: held-out loss per source, HumanEval and
   GSM8K after a short SFT, and the `helix recall` probe at 1B scale. Compare the loss curve against a plain
   transformer of the same size trained by the same engine (`--override index_layer_stride=1000000 --override
   local_blocks=64` turns HELIX into a windowed transformer) for the same tokens. **If HELIX is not within a few
   percent of the transformer's loss at equal compute, stop and fix the architecture; do not scale it.**
3. **3b on 55B tokens, 2 days.** Scaling check: the 1b -> 3b loss improvement should follow a power law; the
   agent SFT on `nebius/SWE-agent-trajectories` should already solve some SWE-bench-Lite tasks.
4. **7b on 153B tokens, 13 days.** The first model that could be useful on its own. Full evaluation.
5. **23b on 467B tokens, ~4 months.** Only with the 7b's numbers in hand and the money in the account.

Each rung is one command with `--hours` set to the time paid for; the engine fits its learning-rate decay to that
time, saves every half hour and pushes to `CorxLabs/Helix-<size>` every hour, so a rung can be paid for in pieces.

## 4. The data

467B tokens, by share of the budget (`PRETRAIN_SOURCES`):

| share | source | what | license |
| ---: | --- | --- | --- |
| 32% | `bigcode/starcoderdata` | permissively licensed source code, 80+ languages, plus GitHub issues, commits and notebooks | The Stack terms (accept once on the Hub) |
| 8% | `codeparrot/github-code-clean` | more permissive code; takes starcoder's share if its terms are not accepted | Apache-2.0 |
| 4% | `OpenCoder-LLM/opc-fineweb-code-corpus` | documentation, tutorials, Q&A about code | MIT |
| 2% | `OpenCoder-LLM/opc-annealing-corpus` | algorithmic corpora and synthetic code QA (two passes) | ODC-BY |
| 36% | `HuggingFaceFW/fineweb-edu` (350BT sample) | educational English web text | ODC-BY |
| 7% | `HuggingFaceTB/finemath` (3+) | mathematics from the web | ODC-BY |
| 4% | `wikimedia/wikipedia` (English) | reference | CC-BY-SA |
| 5% | `HuggingFaceTB/smollm-corpus` cosmopedia-v2 | synthetic textbooks | ODC-BY |
| 1.0% | `jamaica_en` (derived) | English documents about Jamaica, found by `jamaica.is_about_jamaica` while the web and Wikipedia stream past; read up to three times | as above |
| 0.9% | Patois: `fineweb-2/jam_Latn`, `GlotCC jam-Latn`, `MADLAD-400 jam`, Patois Wikipedia | the Jamaican Patois web, a few tens of MB in all, read up to four times | ODC-BY, CC0, CC-BY-SA |
| 0.1% | identity (generated) | who the model is, what CorxLabs is, facts about Jamaica, Patois phrases | - |

Half code and code-adjacent, a third educational web, the rest mathematics, reference and synthetic text, with
Jamaica deliberately over-represented relative to its share of the internet. The Patois corpora are small enough
that their share is a cap on passes (four), not a share of tokens; the honest statement is that the model will have
seen every Patois document on the open web several times, and will speak the language as well as that allows.

Stored as uint16 shards: 467B tokens are 934 GB, which is why the vocabulary is 65,536 and why a 1.6 TB volume
is enough.

**SFT** (`SFT_SOURCES`): resolved SWE-agent runs (`nebius/SWE-agent-trajectories`, `SWE-bench/SWE-smith-trajectories`),
`nvidia/OpenCodeInstruct`, `open-thoughts/OpenThoughts3-1.2M`, `smoltalk2`, `tulu-3-sft-mixture`, and the identity
conversations -- all in one chat format with explicit tool-call and tool-result tokens, loss on the assistant's turns
only, 32k-token sequences.

## 5. Competing with Qwen3.5-27B

Qwen3.5-27B reports 72.4 on SWE-bench Verified, 80.7 on LiveCodeBench v6 and 86.1 on MMLU-Pro. It was trained on
tens of trillions of tokens by a large team with a large cluster, then post-trained at length with reinforcement
learning. A 23B model trained from scratch on 467B tokens -- about 1/80th of the data -- does not reach those numbers
on general benchmarks. Expect, after the full 23B run and SFT, something in the region of the strong open models of
2023-24 on knowledge (MMLU-Pro in the 50s) and better than that on code, because half the data is code and the
fine-tune is agentic. That would be a real, publishable model and a first for Jamaica; it would not be a Qwen3.5 beater.

Where Helix can genuinely compete:

- **Long-horizon agent sessions.** HELIX's cost per token does not grow with the context, and its hot decode state is
  fixed. On agent tasks that run to hundreds of thousands of tokens, a 23B HELIX can serve where a 27B transformer
  cannot, and the training ran at 8k-32k context without paying the quadratic bill.
- **Exact recall over a repository.** The index strand retrieves exact tokens from anywhere in the past; the recall
  probe shows the mechanism works where recurrent models fail. Measured on long-context agent benchmarks, that is the
  claim to test.
- **Jamaican Patois and Jamaica.** No frontier model was trained with this data deliberately up-weighted.

The evaluation (`helix eval`) prints Helix's numbers beside the published targets, and `--baseline` re-runs a target
with the same harness when it fits on the machine, so the comparison is like for like.

## 6. Run order on the server

```
bash helix_setup.sh                     # volume, venv, package; asks for the HF token and the hours paid for
# it then runs, in a tmux session, resumable at every stage:
helix data verify                        # every source reachable with this token
helix tokenizer sample / train           # the CorxLabs tokenizer from a 10 GB sample of the mixture
helix data prepare --tokens <budget>     # shards, smallest sources first; the GPU starts when the first are in
helix train --size 350m --hours 0.3      # smoke
helix train --size <rung> --hours <H>    # the rung the hours pay for, pushed to CorxLabs/Helix-<size> hourly
helix sft prepare / helix sft train      # the coding-agent fine-tune from the pretrained export
helix eval --model <export>              # the table
```
