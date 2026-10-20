# ANANSI and helix-lm 2.0: learning faster, thinking deeper, hearing

helix-lm 2.0 is a major upgrade of the engine that trains HELIX, in three parts:

1. **Faster learning per token and per GPU-hour** in the trainer: the Muon optimizer, muP hyperparameter
   transfer, multi-token prediction, progressive growth, and online data-mixture reweighting.
2. **ANANSI**, a new network built around HELIX's three strands: many residual streams mixed by
   manifold-constrained hyper-connections, a looped reasoning core whose depth is a dial at inference, gated
   attention, a hashed n-gram memory, an optional fine-grained mixture of experts, sandwich norms, and a
   recurrence-only prelude. Named after the spider of Jamaican and Akan stories: it weaves.
3. **Audio as a first-class modality**: a raw-waveform codec at 25 frames a second with a split residual
   quantizer, a speaker encoder, score streams for singing, and an omni wrapper that puts text tokens and
   audio frames in one Anansi sequence with Moshi-style acoustic delay and a depth transformer.

Everything runs and is tested on a CPU at toy scale (`pytest`, about ten minutes). Nothing in 2.0 has been
trained at scale here. This document records what each piece is, the published evidence it rests on (three
research passes over the 2024-2026 literature, October 2026), what the toy suite shows on this machine, and
what remains to be proved on real GPUs. Where a number is a claim from a paper it is labelled as such.

## 1. Learning faster

### Muon (`helix_lm.optim.muon`, `helix train --optimizer muon`)

The update of every 2D weight inside the blocks is the orthogonalized momentum: five Newton-Schulz iterations
(coefficients 3.4445, -4.7750, 2.0315) in bf16 turn the momentum matrix into the nearest matrix with all
singular values near one, so every direction a matrix has not yet learned moves as fast as the common ones.
Moonshot's shape-matched scale `0.2 sqrt(max(rows, cols))` makes one learning rate serve Muon and AdamW, and
weight decay is decoupled. Embeddings, the output head and every 1D parameter stay on AdamW.

Evidence: Moonshot, "Muon is Scalable for LLM Training" (arXiv 2502.16982): the same compute-optimal loss at
about 52% of AdamW's FLOPs on a 399M-1.5B scan; Moonlight 16B-A3B on 5.7T tokens. Kimi K2 (2507.20534): a
1T-parameter model trained with Muon on 15.5T tokens without loss spikes, with QK-clip against attention
logit growth (HELIX has QK-norm). Under FSDP2 each matrix's momentum is all-gathered for the orthogonalization
and the rank's shard written back: one extra gather per matrix per step, the layout Moonshot used too. The
split (`partition_parameters`), the singular values, the update RMS and a Chained state-dict round trip are
tested; the FSDP gather path is not exercised on a CPU.

### muP (`helix_lm.optim.mup`, `--mup --mup-base-width 256`)

Per-parameter init and learning-rate rules so that the learning rate found on a small pilot transfers to the
wide model: embeddings keep the base init and lr; hidden matrices get `std / sqrt(m)`, `lr / m` and weight
decay `x m` (so `lr x wd` is fixed); the output head gets `std / m`, `lr / m` and its logits are multiplied by
`1 / m` (`AnansiConfig.output_multiplier`), `m` being the width ratio. With Muon the hidden matrices keep the
base lr, because the orthogonalized update already has width-independent size. Evidence: Yang et al., Tensor
Programs V (2022); Cerebras's form of the rules; u-muP (2407.17465) and CompleteP (2505.01618, 11.8% FLOPs
saved at compute-optimal shape over muP by adding the depth rule, not implemented here). The coordinate check
(`tests/test_optim.py`) shows the activation update of a 512-wide toy model stays within a small factor of a
32-wide one under the plan and grows without it.

Practice: sweep the learning rate on `a-350m`/`1b` with `--mup`, then use the best value unchanged on the big
run with the same `--mup-base-width`. Do not turn `--mup` on for a big run whose lr was *not* found that way:
the hidden-matrix lr would be divided by the width ratio.

### Multi-token prediction (`AnansiConfig.mtp_depth = 1`, `mtp_weight = 0.1`)

DeepSeek-V3's sequential module: the trunk's representation of position `t` and the embedding of token `t+1`
are normalized, concatenated, projected, run through one more HELIX block and scored with the shared head
against token `t+2`. Evidence: Gloeckle et al. 2024 (2404.19737): +12% HumanEval and +17% MBPP at 13B, gains
grow with size, up to 3x faster self-speculative decoding; DeepSeek-V3 (2412.19437): HumanEval 44.5 -> 53.7 at
228B with a small MMLU dip, second-token acceptance 85-90% for 1.8x decode speed. KORMo's 1B replication
found no average gain at that size, so the module is on at every size for the drafter and the weight is kept
at DeepSeek's later value 0.1. `AnansiForCausalLM.draft` returns two tokens from one pass; the trainer logs
`mtp` separately and the held-out loss never includes it.

### Progressive growth (`helix_lm.growth`, `helix grow SOURCE TARGET OUT --mode stack|identity`)

Train small, grow, continue. Two kinds of depth growth: `identity` inserts blocks whose residual writers are
zero (the grown model computes exactly the old function; tested to 1e-5), and `stack`, which copies trained
blocks as they are. The evidence favours stacking: G_stack (2405.15319, NeurIPS 2024) reached a 7B baseline's
300B-token loss at 194B tokens (54.6%) and beat the function-preserving variants; HyperCloning (Apple,
2409.12903) 2.2-4x to baseline accuracy for width. Width growth here covers the SwiGLU hidden size and Anansi's
stream count (both exactly preserving); hidden-width cloning is not implemented. Growth happens on exports, so
it is independent of how the running job is sharded: `helix grow`, then `helix train --init-from`. Reset the
optimizer (a fresh run does), re-warm the learning rate, and grow only in the stable phase of the schedule.

### Online data-mixture reweighting (`--reweight-strength 1.0`)

After every held-out evaluation, a source whose loss fell faster than average since the last evaluation gets
more of the next batches and a stalled one less: `w_i = prior_i (progress_i / mean progress) ^ strength`,
clipped to `[0.5, 2] x prior`, epoch caps re-applied, frozen during the cooldown. This is the simplest member
of the family (DoReMi 2305.10429: 2.6x fewer steps to baseline at 8B with a proxy model; ADO 2410.11820 and
Aioli 2411.05735: proxy-free, matching DoReMi at 124M-1.3B). Expect a small gain on downstream numbers and a
guard against wasting the year on a source the model has already learned. Tested on the loader and in a run.

### Also from the literature, not implemented

Rho-1's selective token loss (needs a reference model; narrows the distribution), token superposition (Nous,
2605.06546; one lab, no replication), Attention Residuals and xHC (alternatives to mHC that share its slot),
and the critical-batch-size laws (Step Law 2503.04715, Power Lines 2505.13738), which are advice for choosing
`--global-batch-tokens` and weight decay rather than code.

## 2. The Anansi network

### Layout

```
embeddings (+ hashed n-gram memory)
  -> prelude: P layers, recurrence + MLP only, run once
  -> core:    C layers, run `loops` times; the prelude's output is injected at every pass with per-pass
              gain and bias vectors; gradients flow through the last `backprop_loops` passes
  -> coda:    Q layers, run once
  -> norm -> output head (and the multi-token predictor)
```

Every layer sits on `num_streams` residual streams instead of one. A block reads a non-negative mixture of
the streams, writes its (sandwich-normalized) output back with weights in `(0, 2)`, and mixes the streams with
a doubly-stochastic matrix (Sinkhorn-normalized in fp32). At initialization block `i` reads and writes stream
`i mod n` and the mixing is the identity, so the network starts as a plain residual network and is free to
re-wire. Each coefficient has a small input-dependent part. This is DeepSeek's manifold-constrained
hyper-connections (mHC, 2512.24880; adopted in DeepSeek-V4): at 27B on 262B tokens, loss -0.021, BBH 43.8 ->
51.0, GSM8K 46.7 -> 53.8, for +6.7% step time; unconstrained hyper-connections diverged at scale, which is what
the doubly-stochastic constraint fixes. Cost: `n x` residual memory traffic; the blocks themselves are unchanged.

The core's loop count is sampled during training (`min_loops + Poisson(mean - min)`, clipped) from a
counter-seeded generator so every data-parallel rank agrees; evaluation and decoding use the rounded mean, and
`AnansiForCausalLM.think` runs pass by pass and stops when the next-token distribution stops changing (KL below
a threshold). Evidence: Huginn (2502.05171): a 3.5B model with prelude 2 / core 4 / coda 2 reached 7B-class
reasoning scores by looping at test time, with gains concentrated on ARC and GSM8K; Ouro (2510.25741): a 1.4B
looped model at 82.4 MATH500 against Qwen3-4B's 59.6 (at about 2.6x the FLOPs per token), and gains are on
"knowledge manipulation, not storage"; Saunshi et al. (2502.17416): `k` layers looped `L` times match `kL`
layers on multi-hop induction and addition. Three things those papers found non-negotiable are in the code:
input injection every pass, **sandwich norms** (the pre-norm-only variant learned to ignore its state), and a
lower learning rate for looped cores than for plain stacks.

The prelude has **no attention**: a controlled hybrid study (Meta/KAIST, 2510.04800, 350M-3B) found early
attention layers learn uniform maps and that attention belongs mid-depth; the prelude is the delta memory, the
MLP and the n-gram lookup. Attention in the mixer is **gated**: a sigmoid from the block input multiplies the
attention output (bias +2, almost open at start). Evidence: "Gated Attention for LLMs" (2505.06708, NeurIPS
2025): 15B-A2B MoE perplexity 6.03 -> 5.76, first-token attention share 46.7% -> 4.8%, survived learning rates
where the baseline diverged; Qwen3-Next ships it.

The **hashed n-gram memory** adds, for each order 2 and 3, one row of a 65,536-bucket table picked by a
multiplicative hash of the last tokens, projected into the residual width from a zero start. DeepSeek's Engram
(2601.07372) is the scaled version of this idea (hashed n-grams with eight heads, fused at two layers of a 27B,
MMLU +3.0, NIAH 84.2 -> 97.0 at iso-FLOPs); here the tables are small and the fusion is at the input, which is
the cheap end of the design.

The **mixture of experts** option (`num_experts`) turns the core's MLPs into fine-grained routed experts with
a shared expert and DeepSeek-V3's auxiliary-loss-free balancing (a per-expert bias nudged by the sign of the
load imbalance, with load counts all-reduced across ranks). The named sizes are dense; the option exists for
runs with memory to spare, and the toy test shows the balancing acting within one step.

### Sizes

| size | params | active per token | distinct layers | effective depth | TFLOP/token at 8k |
| --- | --- | --- | --- | --- | --- |
| a-350m | 0.36B | 0.47B | 10 | 22 | 0.003 |
| a-1b | 1.10B | 1.86B | 12 | 28 | 0.013 |
| a-3b | 1.99B | 3.66B | 16 | 36 | 0.024 |
| a-7b | 5.46B | 10.8B | 20 | 44 | 0.069 |
| a-24b | 22.8B | 35.5B | 40 | 64 | 0.223 |
| a-34b | 31.6B | 42.1B | 56 | 76 | 0.264 |

"Active" counts the core once per pass at the mean loop count; the FLOPs follow it. `a-24b` costs about 7%
more per token than HELIX `34b` (0.2085 TFLOP) with two thirds of the parameters and the same effective depth;
`a-34b` costs 27% more per token. Loops buy depth per token with parameters held fixed; the year's token budget
shrinks accordingly (`helix size a-24b` prints the hours).

### The toy suite (`helix anansi bench`)

Four tasks that separate architectures in minutes on a CPU: associative recall (MQAR), parity, multi-hop
chain following, and copy after noise. Tiny models of the three families (width 128, four layers of compute,
Anansi as 1 + 2 x 2 loops + 1 with two streams) train for the same steps with Muon and are scored on held-out
batches. The literature predicts: attention and landmarks lead on recall and copy; recurrences with signed
updates lead on parity; looped cores lead on multi-hop. The run on this machine is recorded in the section
"Measured here" below; it is a smoke signal, not a result about 34B models.

## 3. Audio

### What "without blurring" means and how the codec keeps detail

A tokenizer blurs audio in three ways: a low frame rate (Mimi's 12.5 Hz frame is 80 ms, longer than a plosive
burst and half a vibrato cycle, so timing inside the frame becomes an average), too few bits per frame, and
semantic-only tokens that drop speaker and prosody by design. What preserves detail, with the published
numbers: a higher frame rate (WavTokenizer 40 -> 75 Hz: PESQ 1.70 -> 2.37), more residual codebooks (DAC 1 -> 9:
PESQ 1.13 -> about 3.9), split semantic/acoustic streams (Mimi's split RVQ: MUSHRA 57.8 -> 64.0 against plain
RVQ), and continuous features where quantization is not needed.

The codec here (`helix_lm.audio`) follows that evidence: 24 kHz, a causal SEANet encoder with Snake activations
and strides (4, 5, 6, 8) -> **25 frames a second**, a **split residual quantizer** (one semantic codebook
distilled from a speech model when a teacher is given, seven acoustic) with 2048 entries of 8 factorized
L2-normalized dimensions, EMA codebooks, dead-code restarts and quantizer dropout, a mirror decoder, and the
DAC/BigCodec loss recipe (multi-scale log-mel L1 x15, time L1, commitment x0.25, LS-GAN with a multi-period
discriminator plus feature matching). 8 x 11 bits x 25 Hz = 2.2 kbit/s. Expected quality from the published
codecs trained at this rate: PESQ about 2.8-3.0 and speaker similarity about 0.85 on speech after 400-600k
steps on 10-50k hours; singing is weaker until singing and music are in the codec's data. For *listening* the
language model also takes the continuous encoder frames, so understanding never pays the quantization cost.

### Zero-shot voice cloning

In-context: `OmniForCausalLM.clone_prompt` lays out the reference transcript, the reference audio frames and
the target text, and the model continues in that voice; the loss is masked over the prompt in training. That
is how VALL-E 2, CosyVoice and Llasa do it. A speaker embedding (`SpeakerEncoder`, an ECAPA-shaped network with
attentive statistics pooling, trained with additive-angular-margin softmax) is a second, compact conditioning
path and the number cloning is scored with (SIM-o uses WavLM-TDNN embeddings the same way). Honest ceilings
from the literature: single-stream token LMs up to 8B trained on 250k hours sit at SIM 0.57-0.68 and WER 3-5%
on Seed-TTS test-en (Llasa-1B: 3.22 / 0.572); systems that hand the reference to a flow-matching decoder reach
SIM 0.70-0.78. A 1B Anansi on 50k hours should be expected in the first band.

### Singing voice synthesis

The score becomes four per-frame streams summed into the audio frame embeddings: log-F0 in 256 bins over
50-1100 Hz plus unvoiced, the MIDI note plus rest, the lyric phoneme stretched over the note, and a technique
tag (GTSinger's set). The F0 stream is dropped at random in training so the model sings from notes alone when
no curve is given (DiffSinger/VISinger explicit pitch; Prompt-Singer pitch tokens in a decoder-only
transformer). `estimate_f0` is a plain autocorrelation tracker for tests and data preparation. Open singing data
is about 300 hours in total and almost all non-commercial (GTSinger 80.6 h, OpenSinger 50 h, M4Singer 29.8 h,
Opencpop 5.2 h; VocalSet 10.1 h CC BY); the catalogue carries the licences.

### The omni wrapper and the data plan

Text slots and audio slots share one Anansi trunk; only the periphery differs (embeddings, the audio head,
the depth transformer for codebooks 1-7, Moshi's acoustic delay of one frame). That is the design the omni
models converged on (Moshi; Qwen3-Omni; Meta's Mixture-of-Transformers). The known failure is text
regression: Moshi lost 4.6 MMLU points when audio was added without text replay. The plan therefore keeps text
at half of every batch, trains the codec and aligns its frames with the language model frozen first, and adds
audio in a second stage. Data (hours, licence) is in `helix audio sources`: the commercially usable core is
Emilia-YODAS (114k h, CC BY 4.0), MLS (50k h), People's Speech (30k h), Common Voice (22k h, CC0), LibriTTS-R and
LibriSpeech, plus AudioSetCaps and SIFT-50M for instructions. No permissive Jamaican Patois speech corpus exists;
the two finds (a 42.6 h Patois lyrics ASR set and the BBC Caribbean Service clips in the Caribbean Voices
hackathon) are research or competition-licensed. Recording and licensing Jamaican speech is a project of its own.

## 4. Measured here

Toy suite on this machine (a 4-thread CPU), `helix anansi bench --steps 100 --batch-size 16` (AdamW, lr 2e-3,
seed 0), accuracy after 100 steps, with the wall time of each run:

| task | transformer | HELIX | Anansi |
|---|---|---|---|
| recall | 12.2 % (19 s) | 5.5 % (154 s) | 3.6 % (198 s) |
| parity | 52.3 % (12 s) | 52.3 % (85 s) | 52.3 % (113 s) |
| multihop | 7.8 % (19 s) | 4.7 % (149 s) | 0.8 % (199 s) |
| copynoise | 100.0 % (23 s) | 27.1 % (192 s) | 2.5 % (261 s) |

Read it plainly: at 100 steps the plain transformer is ahead on every task, parity is still at chance for all
three, and the recurrent and looped families are 8-10x slower a step on a CPU without their fused kernels. A
200-step run with batch 32 did not finish inside the 90 minutes available here. The suite is a check that the
mechanisms train and that the three families can be compared on equal terms; it is not evidence that Anansi
learns faster, which only the longer runs (`--steps 300` and up, on a GPU, several seeds) can give. The
measured claims that do hold are the exact ones in `tests/test_anansi.py`: the chunked loss equals the plain
loss, cached decoding equals the full forward to 1e-6, growth preserves the function, MoE balancing moves the
bias towards under-used experts, and `think()` exits early on an easy prompt.

## 5. What remains unproven

- Nothing in 2.0 has run on a GPU. The Muon FSDP gather, the hyper-connection memory traffic at 6144 width,
  FlexAttention inside a looped core and the depth transformer's throughput are all untested at scale.
- HELIX itself has not been shown to match a transformer at 1B or above; Anansi inherits that question and
  adds its own. The pilot stage of the CorX5.5 program now trains HELIX, the transformer twin and Anansi at 1B
  on the same tokens with the same optimizer before the year is spent; that comparison is the evidence.
- The codec, the speaker encoder and the omni model have been trained only on seconds of synthetic audio.
- Looping multiplies compute per token; whether it beats spending the same FLOPs on more tokens is exactly
  what the pilot must show for this data.

## 6. Using it

```bash
helix size a-24b                       # the Anansi ladder with FLOPs and hours
helix anansi demo                      # build a-20m, adaptive depth on a prompt, cached generation
helix anansi bench --steps 300         # the toy suite, three families, four tasks
helix train --size a-1b --optimizer muon --mup --reweight-strength 1 ...   # the 2.0 trainer path
helix grow runs/a-1b/export a-3b runs/a-3b-start --mode stack              # grow, then --init-from
helix audio sources --commercial       # audio data a product may use
helix audio codec-demo                 # a toy codec overfitting a tone
helix tristream sizes                  # 2.1: TriStream-2 on the Anansi trunk (docs/TRISTREAM.md)
helix music analyze song.wav           # 2.1: the song analyser and the frisson critic
```

`CORX_SIZE=a-24b` or `a-34b` in the CorX5.5 startup script trains Anansi instead of HELIX; `CORX_OPTIMIZER`
picks `muon` (default) or `adamw`; the smoke run falls back to AdamW by itself if Muon fails on the machine.
