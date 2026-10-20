# TriStream-2: a singing voice model that listens to the song first

TriStream-2 is the 765M-parameter singing voice synthesis and conversion model in helix-lm 2.1, built on the
Anansi trunk, together with the music stack that tells it *what* to sing: a song analyser, a frisson ("chills")
critic, a conductor that plans the vocal, a separator with bleed control, an effects chain, data rules against
accent and register blending, and a one-GPU training program for a 96 GB RTX PRO 6000 Blackwell.

Everything in this document is implemented in `helix_lm.music` and exercised at toy scale by
`tests/test_music.py`. **Nothing has been trained at size.** Every number about quality is a plan or a
published result from somewhere else, and is labelled as such. Section 10 lists what cannot be claimed.

---

## 1. The brief, and where each line of it lives

| The ask | The mechanism | Where |
|---|---|---|
| emotion of the voice, the breath | emotion tag, ten continuous sliders (breathiness, whisper, power, tension, vibrato, warmth, echo, room, volume, brightness), breath events found in the data and fed as a noise-stream feature | `regularize.breath_events`, `tristream.source_features`, `conductor.SLIDERS` |
| master at zero-shot cloning | a speaker encoder (ECAPA-style, AAM-softmax) and, more important, **in-context cloning**: training clips are conditioned on their own voice embedding and on masked spans of their own frames, so at inference a 5-10 s reference is the prompt (Voicebox / VALL-E style) | `audio.speaker`, `tristream.TriStreamForSinging.forward` (masked-span editing) |
| clean SVS, no glitches at high notes or power | a rectified-flow decoder over codec frames (no autoregressive sampling to drift), a source-filter sketch that keeps the harmonic structure explicit, and a glitch detector + repair pass on every output | `tristream.velocity/sample`, `source_filter_sketch`, `effects.detect_glitches/repair_glitches` |
| two singers on one song, isolate one voice | band-split separator with target-singer extraction (FiLM on the speaker embedding), so "this singer" is pulled out and the other voice stays in the mix | `separation.BandSplitSeparator.extract`, `conversion.swap_voice(singer_reference=...)` |
| upload a raw song + the voice to switch to | the swap pipeline: separate, analyse, capture the vocal's effects, convert, re-apply the effects, place back in the mix, gates | `conversion.swap_voice`, `helix music swap SONG REFERENCE` |
| know the instruments, prevent bleed | mask budgeting (the stems' masks sum to one so energy can't be invented), harmonic gating of the vocal stem on its own F0, a bleed report in dB on every result | `separation.separation_loss`, `harmonic_gate`, `bleed_report` |
| no accent from the training data blending in | every clip carries an accent tag; accents are balanced per batch; the model always sees the tag so at inference the tag is chosen, never inherited; a gradient-reversal accent adversary on the content stream | `regularize.ACCENTS`, `balanced_weights`, `AccentAdversary`, `tristream` (`accent_adv` loss) |
| low notes, high notes | pitch-range buckets (bass .. whistle) balanced in sampling; gaps are reported, not hidden; the melody is fitted to the target voice's range by octaves first and compression last; formant shift keeps the singer's size when the register moves | `regularize.RANGE_BUCKETS`, `report_gaps`, `harmony.fit_to_range`, `effects.formant_shift` |
| echo, muffled, tuned voices, instrument louder than the voice | `augment` turns dry targets into those *input* conditions (room, tempo echo, muffle, autotune-style retune, instrumental up to 2.5x louder, a second singer) while the target stays dry; the swap pipeline measures the room and the retune on the input and re-applies them | `regularize.augment`, `conversion.capture_effects`, `effects.match_reverb/retune` |
| pauses | rests are first-class in the plan (`-1` notes), a beat of silence is planned before every entry, the duration predictor learns note and rest lengths | `conductor.ENTRY_GAP_BEATS`, `tristream` (duration head) |
| multiple file uploads | folders and files, recursive, any of wav/flac/mp3/ogg/m4a, with a manifest and a gaps report | `music.data.find_audio`, `prepare_uploads`, `helix music prepare` |
| musical pleasure / dopamine critic | the frisson critic: per-beat features from the chills literature, prior weights, peaks with reasons, a learnable head | `frisson.*`, section 3 |
| the Actor that instructs the SVS | the conductor: a `VocalPlan` with phrases, notes snapped to the key, harmonies, sliders, interventions and a log of why | `conductor.plan_vocals`, `render_controls` |
| sudden soloist entrance | an instrumental song gets the single strongest moment the critic finds, entered after a rest; a voice entry is the highest-weighted intervention | `conductor` ("sudden soloist" branch), `frisson.INTERVENTION_GAIN["voice_entry"]` |
| beat tracking and downbeats, structure, key, RMS -> delivery, centroid -> warmth, genre cues, beat/instrument manipulation | section 4; `effects.duck`, `echo` on the beat grid, stems out of the separator | `analysis.*`, `conductor.sliders_for` |

---

## 2. The model

### 2.1 TriStream-1, in one paragraph

TriStream-1 (the earlier CorxLabs design this continues) split a singing voice into three streams the way a
source-filter model does: a **source** stream (F0, energy, voicing: what the glottis does), a **filter** stream
(the vocal-tract shape: what makes a vowel a vowel and a singer this singer), and a **residual / noise**
stream (breath, fricatives, air), plus a **lyric** stream from phonemes, fused by a transformer into a
rectified-flow decoder over 100-band mel frames at 24 kHz (layers 3 / 5 / 2 / 4 / fusion 8 / decoder 5, width
768). A gradient-reversal singer head kept the content streams from memorising voices; masked-span editing
was a training task; validation was singer-disjoint; targets had to be dry.

### 2.2 What changes in TriStream-2

| | TriStream-1 | TriStream-2 (`ts-765m`) |
|---|---|---|
| width / heads | 768 / 12 | 1280 / 10 (MLP 3840) |
| source / filter / noise / lyric encoders | 3 / 5 / 2 / 4 layers | 4 / 8 / 3 / 4 layers, each a pre-norm transformer encoder |
| fusion | 8-layer transformer | **Anansi trunk**: 2 prelude + 8 core looped 1.5x on average + 2 coda layers, 2 residual streams mixed by manifold-constrained hyper-connections, gated attention, recurrence-only prelude (the delta-rule strand carries the long musical context cheaply) |
| decoder | rectified flow over mel | rectified flow over **codec frames** (the 2.0 codec, 25 Hz, 256-dim continuous latents), 6 layers, with the source-filter sketch added as a structural prior |
| parameters | ~300M | **764.5M** (`helix tristream sizes`) |
| cloning | speaker embedding | speaker embedding **and** in-context prompting through masked-span editing |
| adversaries | singer | singer **and accent** (gradient reversal, scale 0.1) |
| auxiliary heads | - | pitch (256 bins, 0.3), duration (0.1) |
| controls | F0, phonemes | F0, energy, breath, vibrato, 10 sliders, emotion and accent tags, technique |

The losses: `total = flow + 0.3 pitch + 0.1 duration + 0.1 singer_adv + 0.1 accent_adv`. The flow loss is
the rectified-flow velocity regression (Liu, Gong & Liu 2022; Lipman et al. 2022) restricted to the masked
span, so the model learns to *edit* a vocal given its surroundings, which is exactly the conversion task
(keep the content and timing, replace the voice). Inference is 16 Euler steps by default.

Why codec frames instead of mel: the 2.0 codec (`docs/ANANSI.md` section 3) was built to keep breath and
consonant detail that mel + vocoder blur; the continuous latents (passthrough rate 0.25 during codec training)
give the flow decoder a target without quantisation error, and the codec's own decoder renders waveforms.
Published anchors for this route: DAC (Kumar et al. 2023) at 44.1 kHz, Mimi's split semantic / acoustic
quantiser (Défossez et al. 2024), and flow-matching speech models that edit masked spans (Voicebox, Le et
al. 2023; F5-TTS, Chen et al. 2024). If the in-house codec is not good enough after its 240 GPU-hours,
DAC-44k (MIT) fine-tuned on singing is the fallback the molab program is written to accept.

### 2.3 The source-filter sketch

`source_filter_sketch(f0, energy, filter_hidden)` renders a harmonic comb from the F0 stream, shapes it with a
spectral envelope predicted from the filter stream, and adds a noise floor from the noise stream, in the
codec's frame space. It is added to the decoder's output as a residual prior: the network only has to learn
the difference between a textbook source-filter voice and a real one. This is the NSF / DDSP idea (Wang,
Takaki & Yamagishi 2019; Engel et al. 2020) and the reason high notes and belts do not fall apart: the
harmonics are placed by arithmetic, not guessed.

---

## 3. Frisson: what the literature says and how the critic uses it

### 3.1 The findings

* **Sloboda (1991)**, *Psychology of Music* 19(2): 83 listeners located the passages that gave them shivers,
  tears, or a lump in the throat. Shivers clustered on **new or unprepared harmonies** and **sudden dynamic
  or textural changes**; tears on melodic appoggiaturas and sequences; a racing heart on acceleration to a
  cadence and syncopation.
* **Panksepp (1995)**, *Music Perception* 13(2): chills were most often reported for **sad** music and at the
  moment a **solo voice emerges from the background** or a crescendo peaks; he framed the chill as a
  separation-distress response to a "lost" voice. This is the "sudden soloist".
* **Grewe, Nagel, Kopiez & Altenmüller (2007)**, *Music Perception* 24(3): chills with skin-conductance
  responses coincided with the **entry of a voice or instrument**, the **beginning of a new section**, and
  **volume peaks**; chills were rare (a few percent of listening time), differed between listeners, and were
  not simple reflexes to loudness.
* **Guhn, Hamm & Zentner (2007)**, *Music Perception* 24(5): chill passages shared a **crescendo**, an
  **expansion of the frequency range** (a solo instrument or voice entering in a high register above the
  texture), and alternation between solo and accompaniment; heart rate and skin conductance rose at those
  points.
* **Huron (2006)**, *Sweet Anticipation*: frisson is the body's fast surprise response (the ITPRA model)
  followed by a positive appraisal, which is why the strongest chills need **expectation set up and broken** a
  little: a predictable context makes the surprise land.
* **Salimpoor, Benovoy, Larcher, Dagher & Zatorre (2011)**, *Nature Neuroscience* 14: dopamine release in the
  caudate during the **anticipation** (the seconds before the peak) and in the nucleus accumbens at the
  **peak** itself; distinct pathways for the build-up and the arrival. Blood & Zatorre (2001, *PNAS*) had
  shown reward-circuit activity scaling with chill intensity.
* **Bannister & Eerola (2018)**, *Frontiers in Psychology* 9: removing a crescendo, a sudden dynamic change,
  or an instrument's entry from pieces **reduced** the chills they caused: the first causal evidence for the
  triggers. **Bannister (2020)**, *Music & Science*: raising a piece's loudness raised chill reports;
  brightness changes did less.
* **de Fleurian & Pearce (2021)**, *Psychological Bulletin* 147(9), a systematic review of 150+ studies, and
  their corpus work: chills pieces are on average slower, sadder and lower in valence, and in timestamped
  chill onsets **vocals and voice entries are over-represented**.
* **Cheung et al. (2019)**, *Current Biology* 29(23) and **Gold et al. (2019)**, *Journal of Neuroscience*
  39(47): pleasure follows an **inverted U in surprise**: a moderately unexpected chord in a low-uncertainty
  context (or an expected chord in a high-uncertainty one) is liked most.
* **Witek et al. (2014)**, *PLoS ONE* 9(4): the **groove** inverted U: medium syncopation produces the most
  wanting-to-move and pleasure.
* **Solberg & Dibben (2019)**, *Music Perception* 36(4): in electronic dance music the **break routine**
  (breakdown, build-up, drop) produces the peak; skin conductance rises through the build and the **dip or
  silence right before the drop** sharpens it.
* **Juslin (2013)**, *Physics of Life Reviews*: the BRECVEMA mechanisms (brain-stem reflex, rhythmic
  entrainment, contagion, expectancy, aesthetic judgement, ...) explain why no single feature predicts chills:
  the critic therefore combines several.
* Individual differences: openness to experience (Nusbaum & Silvia 2011), white-matter connectivity between
  auditory and reward regions (Sachs et al. 2016). A population score cannot predict one listener.

### 3.2 What the critic computes

Per beat, from `analysis.analyze`: `surge` (RMS step over the previous bars), `crescendo` (slope of the last 8
beats), `entry` (a new spectral component appearing: new voice or instrument), `range` (expansion of the
occupied frequency band upward), `surprise` (chord transition improbability against the song's own
transition counts, pushed through the inverted U), `resolution` (tension falling after a high), `silence`
(a dip right before this beat), `groove` (medium syncopation), `downbeat`, `chorus_start`.

The score is a logistic over the weighted sum. The weights are **priors** in log-odds units, ranked by the
strength of evidence (causal > physiological > self-report):

| feature | prior | the evidence behind the rank |
|---|---|---|
| surge | 1.8 | Bannister & Eerola 2018 and Bannister 2020 (causal); Grewe 2007; Guhn 2007 |
| entry | 1.5 | Panksepp 1995; Grewe 2007; Guhn 2007; de Fleurian & Pearce corpus |
| surprise | 1.2 | Sloboda 1991; Huron 2006; Cheung 2019; Gold 2019 |
| range | 1.0 | Guhn 2007 |
| resolution | 0.8 | Sloboda 1991 (cadence); Huron 2006 |
| crescendo | 0.7 | Guhn 2007; Salimpoor 2011 (anticipation) |
| silence | 0.7 | Solberg & Dibben 2019 |
| chorus_start | 0.6 | Grewe 2007 (new section) |
| groove, downbeat | 0.4 | Witek 2014; Sloboda 1991 |
| bias | -3.0 | chills are rare |

`FrissonHead` is a small network over the same features that **replaces** the priors once labels exist: chill
timestamps (the de Fleurian corpus lists onsets but not audio; your own listening tests do), or continuous
arousal (DEAM, PMEmo, both in the catalogue). `test_critic_and_conductor_make_a_sane_plan` fits it on the
prior's own labels as a smoke test. Until then every output is marked as a prior.

### 3.3 Timing rules the conductor obeys

* Planned peaks at least **20 s apart** (anticipation of roughly 15 s before a peak, Salimpoor 2011, plus a
  chill of several seconds).
* A pre-chorus **ramp of 4-8 bars** (crescendo, rising line) because the build is where the anticipatory
  dopamine is.
* **One beat of rest** before an entry (Solberg & Dibben's dip; Panksepp's voice "emerging").
* The **climax figure**: a leap of an octave (or more) to a held top note that is a chord tone, 1-2 beats
  after the downbeat (Guhn's frequency-range expansion on a resolution).
* **Restraint**: `score_plan` penalises density, off-grid notes, out-of-key or out-of-range notes and repeated
  tricks; of several candidate plans the best *natural* one wins; interventions are capped per minute.

---

## 4. Hearing the song

All of `helix_lm.music.analysis` is plain PyTorch, no external audio library, so it runs anywhere the model
runs.

* **Onsets, tempo, beats, downbeats**: spectral-flux onset envelope, autocorrelation tempogram with a
  log-normal prior around 120 BPM, Ellis's dynamic-programming beat tracker (Ellis 2007), downbeats from the
  bar-period autocorrelation of onset strength plus the chord-change rate. On the synthetic songs in the tests
  the tempo is within a few percent and beats sit on the grid to 80 ms.
* **Key and scale**: chroma from a constant-Q-like filterbank, correlated with the Krumhansl-Kessler major and
  minor profiles (Krumhansl 1990); the result is a tonic, a mode and a confidence, and `snap_midi_to_scale`
  keeps every planned note inside it.
* **Chords and tension**: 24 triad templates per beat, a tension curve from sensory roughness (Plomp &
  Levelt 1965) and distance from the tonic, and a chord-transition surprise from the song's own statistics.
* **Structure**: Foote's novelty on a self-similarity matrix of beat-synchronous features (Foote 2000),
  segments labelled intro / verse / build / chorus / drop / bridge / outro from their loudness, density and
  repetition.
* **Loudness and delivery**: RMS per beat decides breathiness and whisper (quiet) versus power, tension and
  vibrato (loud); the pre-chorus slope decides the build.
* **Brightness and warmth**: spectral centroid; a bright track makes the voice warmer (formant ratio below one),
  a dark track brighter, so the voice sits in the spectrum's gap instead of fighting it.
* **Room**: a decay-time proxy from the energy envelope's tail, so the output can carry the same room.
* **Groove**: syncopation strength and micro-timing; **vibe cues** ("ethereal", "aggressive", "warm acoustic",
  "club", ...) from brightness, density, reverb and tempo, which the conductor turns into slider offsets
  (ethereal: more echo, longer tails, more breath; aggressive: dry, short, more power).
* **Manipulation**: with the stems out of the separator the instruments can be ducked under the voice
  (`effects.duck`), echoed on the beat grid (`effects.echo` at a tempo-locked delay), or re-balanced before
  the voice is placed back (`place_in_mix`).

---

## 5. Separation, duets and bleed

`BandSplitSeparator` follows the band-split family (Luo & Yu 2023; Lu et al. 2023, BS-RoFormer): the STFT is cut
into bands, each band gets its own projection, a transformer runs across time and across bands, and a mask
per stem is produced. Two things are specific to this brief:

* **mask budgeting**: the stems' masks are a softmax, so they sum to one in every bin; a separator cannot
  add energy that was not in the mix, which is where most audible bleed artefacts come from;
* **harmonic gating**: the vocal stem is gated by a comb on its own F0 track, so instrument energy between
  the voice's harmonics is attenuated before conversion (`harmonic_gate`), and `bleed_report` prints the
  leakage of the instrumental into the vocal in dB so a gate can refuse a bad separation.

For duets `extract(mix, speakers)` conditions the mask on speaker embeddings with FiLM, so the referenced
singer is pulled and the other voice is left in place; `swap_voice(singer_reference=...)` uses it.

What to ship: the band-split model is small and trainable on MUSDB18-HQ / MoisesDB **for evaluation only**
(both are non-commercial); for a product the pipeline calls **HT-Demucs** (MIT, Rouard et al. 2023) when it is
installed (`pip install "helix-lm[music]"`), which `try_pretrained()` finds, and which separates vocals at about
9 dB SDR on MUSDB18-HQ. Community BS-RoFormer weights score higher but carry no licence and are not used.

---

## 6. Effects and glitch control

`effects.py`: reverb (exponential-decay noise impulse, wet/dry), tempo echo with feedback, muffle (one-pole
low-pass blend), retune to a scale with a speed (0 = untouched, 1 = hard autotune), vibrato, formant shift
(cepstral envelope warped, harmonics kept), doubling, de-essing, ducking, a look-ahead limiter, breath
insertion from a recorded or synthetic breath, and `detect_glitches` (clicks: a sample step far above the
local sample-to-sample motion; dropouts: sudden silence after loud audio; clipping; DC offset) with
`repair_glitches` (linear ramps over clicks, cross-fades over dropouts). The swap pipeline runs the detector
before and after repair and the report carries both counts.

---

## 7. Data

### 7.1 The catalogue (`helix music sources`)

| source | kind | hours | licence | role |
|---|---|---|---|---|
| VocalSet | singing, 20 pro singers, techniques | 10 | CC BY 4.0 | train |
| Dagstuhl ChoirSet, ESMUC choir | choir, close mics, F0 | 1.5 | CC BY 4.0 | train |
| vocadito | singing excerpts with notes and lyrics | 0.3 | CC BY 4.0 | eval |
| Emilia-YODAS | speech, 114k h | 1,000 used | CC BY 4.0 | train (source-filter prior, breath, accents) |
| Slakh2100 | synthesized multitrack, no vocals | 145 | CC BY 4.0 | train (separator negatives, instruments) |
| FMA (CC-BY tracks) | music | ~2,000 | per-track CC | train (analysis, critic) |
| GTSinger, OpenSinger, M4Singer, Opencpop, PopCS, NUS-48E, SingStyle111, PopBuTFy | singing | ~240 | non-commercial | **research only** |
| MUSDB18-HQ, MoisesDB, MedleyDB, MedleyVox | stems, duets | ~32 | non-commercial | **evaluation only** |
| DEAM, PMEmo | arousal / valence labels | 35 | CC / MIT | critic training |
| your uploads | the voices you own | - | yours | the main lever |

The honest total of **commercially usable clean singing in the open is under 100 hours**. That is why the
program (a) pre-trains the source-filter machinery on 1,000 h of speech, (b) runs many augmented epochs over
the singing (each epoch sees different rooms, echoes, instrumentals and duet partners, so it is not
repetition), and (c) makes the uploads folder the first-class source: every hour of consented, dry, tagged
singing you record is worth more than any public set.

### 7.2 The rules (`helix_lm.music.regularize`)

Accent tags on every clip (`unknown` allowed, counted, and reported when it is more than half); per-batch
balancing of accents and of pitch-range buckets with floor 0.25x and ceiling 4x of natural share; a dry-target
check (reverb proxy, roll-off) that rejects wet or band-limited targets (TriStream-1's rule); singer-disjoint
validation by hashing singer ids; breath events from quiet, flat, unvoiced frames before voiced onsets;
`augment` for the input conditions (section 1). The gaps report says which registers and accents the data
cannot teach, so the model card can say it too.

### 7.3 Uploads

`helix music prepare FILES_OR_FOLDERS --out DIR --singer NAME --accent TAG [--mixtures]`: finds every audio
file recursively, resamples, measures F0 range and breaths, checks dryness, separates full songs when
`--mixtures` is given and a separator is available, writes `manifest.json` and the gaps. The same function is
stage `prepare` of the molab program, with `<root>/uploads` as the folder.

---

## 8. The 96 GB program (`helix tristream molab`, `python -m helix_lm.scripts.tristream_molab`)

One file, eight stages, every stage resumable from a marker, a `stop` file to end a run cleanly:

| stage | what | default budget |
|---|---|---|
| `setup` | packages, GPU check, token (from `$HF_TOKEN` or `/root/hf_token`) | minutes |
| `data` | Zenodo downloads (VocalSet, Dagstuhl, ESMUC, Slakh), Emilia-YODAS archives (`--emilia-hours`), research sets only with `--include-research`, size cap `--max-gb` | hours, network-bound |
| `prepare` | separate, resample, F0, breaths, dryness, tags, manifests; your `uploads/` first | hours, CPU-bound |
| `codec` | the 2.0 codec at 24 kHz with the multi-period discriminator, continuous passthrough 0.25 | `--codec-hours 240` |
| `train` | TriStream-2 with Muon, singer-disjoint split, accent balancing, in-context speaker conditioning, checkpoints every 30 min, Hub push with each checkpoint | `--train-hours 200` |
| `eval` | F0 RMSE in cents, voicing error, speaker similarity on held-out singers, glitch rate, accent-probe accuracy | minutes |
| `push` | `CorxLabs/TriStream-2-765M` (or `--hf-repo`), model card with the gaps report | minutes |
| `swap` | `--song` + `--reference` demo through the pipeline | minutes |

**Memory.** 765M parameters in bf16 with fp32 master weights and Muon momentum is about 8 GB; activations for
12 s clips (300 codec frames) are small per clip, so the batch is what fills the card. `--batch 0` (the
default) starts at 64 clips and halves on the first out-of-memory until it fits, which is how the 96 GB is
pushed to its edge without a hand-tuned number; `--clip-seconds` trades clip length for batch.

**Throughput (planned, not measured).** An RTX PRO 6000 Blackwell is inferred at roughly 500 dense bf16
TFLOPS. At 35 % utilisation the trunk + decoder + encoders process on the order of 15-20k codec frames a
second, that is 600-800 s of audio a second, or well over 1,000 hours of audio a day. The run is therefore
**data-bound, not compute-bound**: the whole clean-singing corpus passes in minutes, and the 200 hours are
spent on augmented epochs and the speech prior. The codec stage is the compute-heavy one (it runs at sample
rate with a discriminator), hence its larger budget.

**What to expect after the budget.** A working codec, a model that converts and edits dry vocals in the
registers and accents the data covers, and a gaps report naming the rest. It will not match Seed-Music or
commercial SVS on breadth: they train on orders of magnitude more licensed singing. It can match or beat them
on *your* voices if those voices are in the uploads folder.

---

## 9. Evaluation (`stage eval`, `tests/test_music.py`)

* **pitch**: F0 RMSE in cents and voicing error against the control track, on held-out singers;
* **voice**: cosine similarity between the speaker embedding of the output and of the reference, versus the
  similarity to the *source* singer (conversion must move it);
* **cleanliness**: clicks and dropouts per minute from `detect_glitches`, clipping fraction, leak dB from
  `bleed_report`;
* **accent leakage**: a probe classifier's accuracy on the content stream (lower is better, chance is the
  goal) and a listening check that an `unknown` accent tag yields the reference's accent, not the corpus's;
* **range**: the same metrics split by range bucket, so a weak whistle register is seen, not averaged away;
* **the critic**: AUC of the prior score against DEAM arousal peaks, and against your own chill timestamps
  once collected;
* **listening**: comparative MOS on naturalness, similarity and "chills" between the plan with and without the
  conductor's interventions (the only test that measures the thing itself).

---

## 10. What cannot be claimed, and the ethics

* **No model has been trained.** Sizes, losses, pipelines and tests run at toy scale on a CPU; every quality
  statement above is a plan or someone else's published number.
* The critic predicts a **population tendency** from literature priors. It is not a dopamine meter, it does
  not know any individual listener, and its weights are hand-set until it is fit on labels.
* **Data**: under 100 hours of commercially usable clean singing exist in the open; the research sets are
  non-commercial and the program keeps them out unless `--include-research` is given for evaluation.
* **Separation**: only HT-Demucs (MIT) is cleanly shippable; the in-house separator is a research baseline.
* **Consent**: clone only voices you own or have written permission for; the uploads manifest records the
  singer id and the consent field is yours to fill. Mark generated audio (a watermark such as AudioSeal,
  checking its licence, or a spread-spectrum mark) before anything leaves the lab, keep the plan and the report
  with every output, and do not present a converted performance as a recording of the referenced singer.

---

## 11. Using it

```bash
pip install "helix-lm[music]"                      # soundfile + HT-Demucs for separation
helix tristream sizes                              # ts-tiny / ts-300m / ts-765m
helix tristream demo --steps 10                    # forward, backward, edit, pitch head at toy size
helix music sources --commercial                   # data a product may use
helix music analyze song.wav --top 5               # tempo, key, sections, chills candidates and why
helix music plan song.wav --low 55 --high 79 --out plan.json
helix music swap song.wav reference.wav --out out.wav [singer_to_replace.wav] [--root /workspace/tristream]
helix music prepare ~/takes --out /workspace/tristream/uploads --singer nathan --accent jamaican

# the 96 GB program, in a notebook cell:
!pip install -q "helix-lm[music]"
!python -m helix_lm.scripts.tristream_molab --root /workspace/tristream --stage all --hf-repo CorxLabs/TriStream-2-765M
```

The token for the Hub goes in `$HF_TOKEN` or `/root/hf_token` on the machine, never in a notebook or a chat.
