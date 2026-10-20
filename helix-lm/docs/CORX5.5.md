# CorX5.5: a 34B HELIX model from scratch, one year on 8 x B300

CorX5.5 is CorxLabs' long-horizon coding agent: the HELIX `34b` size (33.87B parameters) trained from a random
initialization on about 870 billion tokens, then fine-tuned into an agent with safeguards. This document is
the plan, the arithmetic, the data, the safeguards and the limits. The program itself is one startup script,
`corx_startup.sh`, which runs every stage below as a systemd service and pushes code, checkpoints and
milestones to the model repository `CorxLabs/CorX5.5` on the Hugging Face Hub.

The decision record, so nobody re-derives it later: the machine is 8 x B300 SXM6 (262 GB each) with a
10 TB volume; the owner chose one year of wall-clock over a shorter run of a smaller model, and HELIX over
a plain transformer, knowing that HELIX has not yet been shown to match a transformer at this scale (the
pilot in stage 6b measures exactly that before the year is spent).

## The arithmetic

| quantity | value |
| --- | --- |
| parameters (`helix size 34b`) | 33.87B (33.06B non-embedding); hidden 6144, 64 layers, 48 heads, 8 KV heads, 24 recurrent heads |
| training FLOPs per token at 8k context | 2.09e11 |
| 8 x B300 dense bf16 | 8 x 2.25 PFLOPS = 18 PFLOPS peak |
| tokens per second at 30 / 35 / 40% MFU | 25.9k / 30.2k / 34.5k |
| 870B tokens at 30 / 35 / 40% MFU | 389 / 333 / 292 days |
| tokens per year at 30 / 35 / 40% MFU | 817B / 953B / 1,089B |

So 870B tokens is one year at about 32% MFU, and `--hours 8760` makes the schedule fit a year whatever the
measured speed: the trainer's warmup-stable-decay schedule is fitted to the deadline, and it stops cleanly
(saves, exports, pushes) when either the token budget or the hour budget is reached. Ninety minutes into the
main run the program prints a `[projection]` line from the measured `tok/s`: that is the real estimate.
Below 25% MFU, look at `helix kernels` first (the fused strands must be on) and at the `--micro-batch` the
OOM back-off settled on.

870B tokens is 26 tokens per parameter: past Chinchilla-optimal (20), far short of what today's open models
see (Qwen3-class models: 36T). CorX5.5 will be a strong 34B model for its token count, specialised in code
and Jamaican text; it will not match Qwen3.5-27B on general benchmarks, which trained on roughly forty times
the data. The comparison table `helix eval` prints keeps the published numbers next to ours.

## Storage, the Hub, the token

| item | size |
| --- | --- |
| prepared data (870B x 1.08 budget x 1.15 overshoot, uint16) | about 2.2 TB |
| one sharded checkpoint (fp32 master weights + Adam, 8 ranks) | about 550 GB; `--keep-checkpoints 2` |
| one bf16 export (what is pushed) | 68 GB |
| Hub repository after a year | latest export + 8 milestones + fine-tune: about 700 GB |

Private repositories over 100 GB need a paid Hub plan (PRO or Team/Enterprise); set `CORX_PUBLIC=1` to make
the repository public, or fund the plan before the first 100B-token milestone (day 40 or so). The token is a
**write** token; the script keeps it only in `/root/hf_token` (mode 600) and never in a log. Four sources are
gated and need their terms accepted once with the same account, otherwise the run proceeds without them and
the code share moves to the ungated `github-code`: `bigcode/the-stack-dedup`, `bigcode/starcoderdata`,
`bigcode/the-stack-github-issues`, `free-law/Caselaw_Access_Project`.

The preparation stops itself when the volume has less than 800 GB free (`--min-free-gb`), so checkpoints
always have room; on a 10 TB volume this never triggers.

## The stages

Everything is resumable: each stage leaves a marker (a file or a log line) and is skipped on the next start.
`journalctl -fu corx` follows the service; `bash /root/corx_stop.sh` stops safely (the trainer saves and
pushes first); `systemctl start corx` continues.

| stage | what | where |
| --- | --- | --- |
| 1 | `helix data verify`: every source reachable with the token; gated ones named | `logs/verify.log` |
| 2 | the CorxLabs tokenizer: 65,536-token byte-level BPE trained on a 16 GB sample of the mixture | `tokenizer/` |
| 3 | `helix jamaica crawl`: Jamaican public records for 12 h in the background (judgments, Hansard, acts, gazettes, statistics, JIS) | `local/jamaica_gov/` |
| 4 | `helix jamaica synth`: 300k SJE/Patois code-switched dialogues from a local Qwen3-32B (Apache-2.0) served by vLLM, then the GPUs are freed | `local/codeswitch*/` |
| 5 | `helix data prepare`: 940B tokens (870B + 8% slack) of shards, in the background, smallest sources first | `tokens/<source>/` |
| 6 | smoke run: HELIX 350m for 25 minutes exercises kernels, FSDP2, checkpoint, export, Hub push | `runs/smoke`, Hub `runs/smoke` |
| 6b | the pilot: HELIX 1b against the transformer twin `t-1b` (same tokens, two hours each) | Hub `runs/pilot-*` |
| 7 | **CorX5.5**: HELIX 34b, 4M-token batches at 8k context, checkpoint every 30 min, push every 12 h, milestone copy every 100B tokens | Hub `checkpoints/corx5.5`, `milestones/` |
| 8 | CorX5.5-Instruct: agent trajectories, domain chats, code-switched dialogues, safety and identity data at 32k context | Hub `instruct/` |
| 9 | benchmarks, the refusal / over-refusal check, the model card | Hub `README.md` |

The main run begins when 15% of the data (130B tokens) is on disk; the loader opens whatever sources exist and
sources prepared later join at the next restart (they start with fresh cursors; nothing is lost). The order of
preparation puts the small, rare sources first (Patois, Jamaican records, legal, health, math) so they are in
the mixture from the first step.

### The pilot gate

Stage 6b prints `PILOT: mean held-out loss HELIX x vs transformer y`. The twin has 15% more parameters
(1.35B vs 1.17B: the transformer family keeps `head_dim` 128 and a wider MLP) and sees the same tokens.
Within about 2%, HELIX is healthy and the year is spent on it. Clearly worse (5% or more), stop and think:
the same script trains the transformer twin at 34B with `CORX_SIZE=t-34b` (33.9B parameters, Qwen3 layout
with QK-norm, exported as a standard `Qwen3ForCausalLM` that every inference server loads). The program
does not switch by itself: that call is the owner's.

## The data

Pretraining mixture by kind (the full list with repositories is `helix_lm.data.catalog.PRETRAIN_SOURCES`):

| kind | share | sources |
| --- | --- | --- |
| web | 40.0% | DCLM-baseline, FineWeb-Edu, FinePDFs |
| code | 30.0% | The Stack dedup (permissive), StarCoderData, GitHub code, OpenCoder code web, CommitPackFT commits, GitHub issues, execution traces rendered from OpenCodeInstruct and CodeFeedback, OpenCodeReasoning |
| Jamaica | 11.7% | documents about Jamaica found while streaming the web (`jamaica_en`), the wider Caribbean (`caribbean_en`), the crawled public record (`jamaica_gov`), the synthesized code-switched transcripts, the identity documents |
| math | 6.0% | FineMath, InfiWebMath, NuminaMath, Nemotron-Math |
| education | 4.5% | Cosmopedia v2 and v1, WebInstruct |
| Patois | 2.3% | FineWeb-2 `jam_Latn`, GlotCC, MADLAD-400, Jamaican Patois Wikipedia (capped at 3-4 epochs) |
| synthetic | 2.0% | Nemotron Pretraining Specialized |
| legal | 1.5% | Caselaw Access Project |
| health | 1.0% | MedMCQA, PubMedQA, medical-o1, clinical guidelines |
| finance | 1.0% | Finance-Instruct, Sujet Finance |

Structured rows (problem/solution pairs, execution traces with their test results, commits, issues,
multiple-choice questions, guidelines, transcripts) are rendered into plain documents by
`helix_lm.data.renderers` before tokenization, so the model reads them as text, not JSON.

The fine-tune (`SFT_SOURCES`): SWE-agent and SWE-smith trajectories (32%), OpenCodeInstruct (18%),
OpenThoughts3 (12%), tulu-3 (10%), smoltalk2 (8%), the code-switched dialogues (7%), finance and medical
chats (6%), safety (3%), CoCoNot refusals (2%), identity (2%).

### Jamaican data: what is real and what is a caveat

- **Patois text on the open web is small.** The four Patois web sources together hold on the order of a few
  hundred million tokens; they are capped at 3-4 epochs rather than repeated for a year. Most of the model's
  Jamaican register therefore comes from `jamaica_en` (Standard Jamaican English about Jamaica: news,
  government, business), the crawled public record, and the synthetic dialogues.
- **The crawler is polite and public-only.** It honours `robots.txt` and crawl delays (2 s per host), reads
  sitemaps, stays on the listed government and court hosts, and converts HTML and PDF to text. Twelve hours at
  that pace yields tens of thousands of pages, not millions; raise `CORX_CRAWL_HOURS` to crawl longer.
- **Synthetic dialogues are only as Jamaican as the teacher.** Qwen3-32B writes acceptable Patois with the
  style guide in `helix_lm.data.synth.STYLE_GUIDE`, and `acceptable()` rejects dialogues that do not switch
  registers or contain no Patois markers, but a native reader should sample `local/codeswitch-chat/` before
  stage 8 and delete what reads wrong. Nothing from a closed-vendor model is used.
- **Legal and health texts are American and international**, not Jamaican, except what the crawler finds:
  judgments of the Supreme Court and the Court of Appeal, Hansard, the Acts. The safeguards' referrals
  (Legal Aid Council, 110/119, 888-NEW-LIFE, JDIC, MOCA) are Jamaican.

## Safeguards

The safeguards are a layer of training data and a deployment recipe, not a filter inside the weights.

- **Policy** (`helix_lm.data.safety.POLICY`): the system prompt the fine-tune trains against and the one to
  deploy with. It declines help with serious harm, keeps professional boundaries, and answers in SJE or Patois
  to match the person.
- **Refusal data**: CoCoNot and tulu-3's safety subset supply the harmful prompts and the refusals; no harmful
  prompts are authored in this repository. `BOUNDARIES` adds legal, medical and financial questions answered
  with limits and Jamaican referrals; `LOOK_ALIKES` are benign questions that sound alarming (killing a
  process, SQL injection for an exam, paracetamol doses, ethical hacking) answered helpfully, so the model
  learns the difference; `AGENT_CONDUCT` covers what a coding agent must not do unasked (delete, force-push,
  disable tests, exfiltrate secrets).
- **Measurement**: `helix eval` runs XSTest (should-comply prompts: over-refusal rate) and a harmful set
  (refusal rate) with `looks_like_refusal`; both numbers go in `eval.json` and the model card.
- **Deployment**: run CorX5.5-Instruct behind a moderation model on both input and output
  (Qwen3Guard-Gen-8B, Apache-2.0, is the current recommendation), with the policy as the system prompt, tool
  calls sandboxed, and logs kept. A 34B base model trained on the open web knows things the fine-tune only
  teaches it to decline; the moderation model is the second lock on the door.

## What a real GPU has not seen yet

The engine is tested on a CPU at toy scale (the suite runs in about seven minutes): fused attention equals
the reference, the loader resumes exactly, a run checkpoints and resumes to the same weights, two FSDP2
processes checkpoint and resume, the SFT masking, the HF export of the transformer family. Not yet exercised
on hardware: the flash-linear-attention kernel on Blackwell, FlexAttention compile times at 8k context for
64 layers, the real MFU, vLLM serving Qwen3-32B on B300 for stage 4, Hub uploads of 68 GB exports, and the
throughput of the preparation on the machine's network. The smoke run (stage 6) is designed to surface the
first four within half an hour; the program stops there if it fails rather than continuing blind.

Multi-node (`CORX_NODES` > 1) is wired through `torchrun` but assumes the data directory is on shared
storage; it has not been run.
