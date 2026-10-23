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
| parameters | ~300M | **765M** (`helix tristream sizes`; 765.1M with the 2.2 additions) |
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
| `codec` | the 2.0 codec at 24 kHz (or `--codec-rate 50`) with the multi-period discriminator, continuous passthrough 0.25 | `--codec-hours 240` |
| `refiner` | the sample-rate refiner on (codec reconstruction, original) pairs: phase-true harmonics, 10 ms noise, snake residual, transient and high-band losses, discriminator after 2,000 steps | `--refiner-hours 48` |
| `air` | the air band above the codec's Nyquist from your 48 kHz uploads (opt in) | `--air-hours 0` |
| `train` | TriStream-2 with Muon, singer-disjoint split, accent balancing, in-context speaker conditioning, checkpoints every 30 min, Hub push with each checkpoint | `--train-hours 200` |
| `eval` | through the fail-safe loop on held-out singers: F0 RMSE in cents, speaker similarity, glitch rate, frame-rate buzz, HNR, sync lag, robotic score, rounds, fallback rate, pass rate | minutes |
| `push` | `CorxLabs/TriStream-2-765M` (or `--hf-repo`), model card with the gaps report | minutes |
| `swap` | `--song` + `--reference` demo through the pipeline | minutes |

**Speech against singing: staging and balance.** Speech is in the mix for the source-filter machinery
(breath, consonants, the tract), not for the voice's character, and a thousand hours of it would swamp
fifteen hours of singing in any sampler that draws by clip count. So the sampler fixes the **domain
share first**: `regularize.mixture_weights` gives singing (with choir) `1 - speech_share` of every batch
and speech `speech_share`, whatever the corpus sizes, and only then balances pitch-range buckets and
accents inside each domain. The share follows a **curriculum** (`speech_share_at`): 50 % at the start,
annealed linearly to 10 % by the end of the training budget, so the late training, where the voice's
character is set, is sung. Spoken clips also carry a **loss weight** of 0.5 (`batch["weights"]`, applied
to the flow and pitch losses inside `TriStreamForSinging.forward`). The codec draws sung and spoken
files half and half. Evaluation uses sung clips only. The default speech download is 300 h
(`--emilia-hours`); more speech does not change the batch shares, only the variety of the speech half.
The flags are `--speech-share-start 0.5 --speech-share-end 0.1 --speech-loss-weight 0.5`.

**Resume from the Hub.** The program mirrors its state to a private Hub repo (`--hf-repo`, default
`CorxLabs/Tristream-V2-SVS`): checkpoints with a sidecar of step and hours, markers, manifests, eval and
a model card, every `--push-every-hours` (2 by default) and on any failure, interruption or stop file; the
repo history is squashed after each push so the Hub holds one copy of each file. On start the program
pulls whichever checkpoint has more steps than the local one. Budgets carry across restarts: a stage
stopped at hour 30 of 48 resumes with 18 left. A crashed stage is retried (`--retries`, `--retry-wait`).

**Downloads that are whole.** Every file is checked against the size and md5 the publisher lists before it
gets its final name. A body cut short by the server or a proxy is resumed with a Range request; a wrong one
is deleted and fetched again over Zenodo's second route. Large files come down over sixteen ranged connections at once, each piece
resuming on its own, because Zenodo throttles a single connection. An archive that fails its zip or tar check is
deleted, never half-extracted. The data stage insists on every wanted source for `--data-attempts` runs (5)
before going on without one, and a source that arrives later is prepared on the next run. One run per root:
a second process (a notebook cell restarted while the first still runs) refuses to start while `run.lock`
names a live pid.

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

## 11. Using it (see also section 12 for the fail-safe loop)

```bash
pip install "helix-lm[music]"                      # soundfile + HT-Demucs for separation
helix tristream sizes                              # ts-tiny / ts-300m / ts-765m
helix tristream demo --steps 10                    # forward, backward, edit, pitch head at toy size
helix music sources --commercial                   # data a product may use
helix music analyze song.wav --top 5               # tempo, key, sections, chills candidates and why
helix music plan song.wav --low 55 --high 79 --out plan.json
helix music swap song.wav reference.wav --out out.wav [singer_to_replace.wav] [--root /workspace/tristream]
helix music prepare ~/takes --out /workspace/tristream/uploads --singer nathan --accent jamaican

# the 96 GB program, in a notebook cell (resumes from the private Hub repo if the notebook dies):
!pip install -q "helix-lm[music]"
!python -m helix_lm.scripts.tristream_molab --root /workspace/tristream --stage all \
    --hf-repo CorxLabs/Tristream-V2-SVS --size ts-765m --emilia-hours 300 \
    --codec-hours 240 --refiner-hours 48 --train-hours 200 --push-every-hours 2
!python -m helix_lm.scripts.tristream_molab --stage eta      # the time the budgets add up to
```

The token for the Hub goes in `$HF_TOKEN` or `/root/hf_token` on the machine. Every stage creates its own
folder, resumes from its own checkpoint and from the Hub's, and keeps the hours it has already used.

---

## 12. Hardening (2.2): what if it fails

Three failure modes were named against the 2.1 design, and each is real. This section says what was built
against each, what detects it when it happens anyway, and what the system does then. Everything below is in
`helix_lm.audio.refiner`, `helix_lm.music.artifacts`, `helix_lm.music.humanize`, `helix_lm.music.failsafe`
and the 2.2 changes to `tristream.py`, with the tests in `tests/test_hardening.py`.

### 12.1 "Lossy compression": the 25 Hz codec smears consonants, breath and air, and can phase

What is true: a 40 ms frame cannot carry a 5 ms burst or the phase relations between harmonics; the codec's
decoder has to guess them back, and the guess is where metallic phasing and smeared fricatives come from.

What was built:

* **The refiner** (`HarmonicNoiseRefiner`) stops asking the codec to do those jobs. It runs at the sample
  rate after the codec with three paths: a **harmonic oscillator bank** whose phase is accumulated sample by
  sample from a 100 Hz F0 curve (coherent by construction, aliasing excluded by construction), a **noise
  path** shaped by band gains at 100 Hz (a /t/ lands at 10 ms, not 40), and a **snake-activated residual
  network** that corrects what those two cannot express. A learned gate decides how much of the codec's
  output to keep. Its losses are multi-resolution STFT, a **transient loss** on 5 ms spectral-flux curves,
  a **high-band loss** above 6 kHz, time-domain L1 and the multi-period discriminator. The molab program
  trains it in its own stage (`refiner`, 48 h by default) on (codec reconstruction, original) pairs.
* **A 50 Hz codec option** (`CodecConfig.at_frame_rate(50)`, `--codec-rate 50`): half the averaging per
  frame at twice the decoder's sequence length.
* **The flow decoder already works on continuous latents**, never on quantised codes; the quantiser is
  for the language-model path, not for singing.

What detects a failure anyway: `frame_rate_buzz` (the frame rate leaking into the amplitude envelope),
`harmonic_coherence` (harmonics-to-noise along the F0 and its flicker on held notes), `hf_character` (the
band above 6 kHz tonal where the reference is noisy: metallic), `consonant_sharpness` (how much onset
energy survives at the reference's onsets).

What happens then: the loop escalates steps and guidance and re-seeds; if consonants or air still fail it
**transplants the source vocal's own unvoiced high band** (unvoiced bursts carry little identity, so this
is nearly free in similarity and large in crispness); if harmonics still phase or buzz it renders the span
from the **source-filter sketch** (`render_sketch`), which has no learned parameters and cannot phase.

### 12.2 "Cross-stream desynchronization": the streams slip against each other in runs and belts

What is true: the lyric stream is the only one not on the frame grid, and a model that reads pitch at 25 Hz
cannot represent a 60 ms glide or a 6 Hz vibrato faithfully; the classic result is a formant track that
follows the pitch (the "chipmunk" or the robot).

What was built:

* **Sub-frame F0**: the source stream reads the F0 at four points inside every frame (100 Hz) as offsets from
  the frame's value (`source_features(..., f0_fine=...)`, 11 channels), and the refiner's oscillator follows
  the same 100 Hz curve, so a run is a run and not its average.
* **A monotonic alignment prior** on the lyric cross-attention (`alignment_bias`): with the durations
  (ground truth in training, the duration head's prediction at inference) a frame can only attend near the
  phoneme that is sung there. The lyric stream cannot drift against the pitch stream because it is not free to.
* **Formant / pitch decoupling** as a loss: on half of every batch the source stream is shifted by up to
  seven semitones and the envelope the model sketches must not move (`formant_invariance`, weight 0.05).
  This is the invariance the source-filter theory promises, enforced rather than hoped for.
* The source, filter and noise streams were already frame-synchronous; the decoder is bidirectional.

What detects a failure anyway: `sync_lag` (cross-correlation between the control events, voicing onsets
and note changes, and the audio's onsets at 10 ms; vibrato is deliberately not an event), and the pitch
error of the output against the control curve.

What happens then: the failing span is regenerated through the editing path with more steps; the sketch
fallback is coherent with the control curve by construction.

### 12.3 "One-block stale indexing": the trunk cannot react mid-block

What is true: the index strand chooses landmarks at block boundaries from the previous block's end. In a
streaming language model that is a one-block lag. In an offline singing model it is a choice, and the
wrong one, because the whole song is known in advance.

What was built:

* **Look-ahead conditioning** (`lookahead_frames = 8`, one block): the trunk's input at frame *t* also
  carries the source and noise streams at *t + 8*. The index decision at a block boundary is made with
  the *next* block's pitch, breath and energy already in hand: the index is one block early, not one
  block stale. The local strand covers the inside of the block as before.
* **An event track** (`event_features`): beat phase, downbeat, section change and loudness surge from the
  song analysis, fed for the current frame and one block ahead. The drop, the pivot and the sudden entry
  are in the conductor's plan before a note is rendered; the trunk is told.
* **Sway sampling and adaptive steps**: the flow's time grid bends towards *t = 0*, where structure is
  decided, and sampling stops once the velocity field has settled (`adaptive=True`), so the budget goes to
  the hard spans.
* **Classifier-free guidance** on the voice and the sliders (`cfg_dropout = 0.1` in training, `guidance`
  at inference): sharper identity and control when a take comes out vague.

### 12.4 "No robotic sound": the humanizer and the ear for it

A synthesizer gives itself away in the pitch curve before anything else. `humanize` renders a note list
the way a larynx does (portamento with an S-shaped glide and overshoot, preparation from below,
vibrato that starts late, ramps in, drifts and grows, a slow wander and a fine jitter, releases that fall
away, attacks and phrase arches in loudness, a breath before each phrase, onsets never exactly on the
grid), at 100 Hz, with named styles (`pop`, `soul`, `classical`, `reggae`, `ethereal`, `aggressive`) and a
seed per singer. The conductor now renders through it by default and picks the style from the song's
vibe cue. `robotic_score` is the ear for the same tells (fine fluctuation on held notes, constancy of the
vibrato extent, instantaneous jumps instead of glides); on the test curves a machine curve scores above
0.5 and the humanized one below 0.1.

### 12.5 The loop (`failsafe.synthesize_with_gates`)

```
generate (steps, guidance, seed)
  -> listen: artifacts, pitch error, voice similarity, sync lag, level
  -> ok? ship
  -> local failure (clicks, dropouts, a smeared onset)? regenerate that span only, through the editing path
  -> otherwise: steps x2, guidance +0.5, new seed, again (up to max_rounds)
  -> still smeared consonants or air? transplant the source's unvoiced high band
  -> glitch repair, then listen again
  -> still failing any sound gate? render the span from the source-filter sketch (it cannot phase or glitch)
  -> still failing, and this is a conversion? ship the source vocal unchanged and say so
  -> limiter, and a report that says which rung was reached and why
```

Nothing that fails its gates is shipped as it is. An undertrained model produces noise; the loop's last rungs
turn that into either a physically coherent sketch of the planned voice or, for a conversion, the original
voice untouched, with the report naming the failure. That is the guarantee against "weird sounds": not
that the model cannot fail, but that a failure never reaches the listener as noise.

The report is attached to every conversion (`Converter.last_report`), printed by the molab `eval` stage per
held-out clip (pass rate, rounds, fallback rate, buzz, HNR, lag, robotic score alongside pitch error and
similarity), and is what a listening test should be read against.

### 12.6 What this does and does not claim

Each mechanism is implemented, runs on a CPU at toy scale, and is tested for the property it was built
for (phase-true oscillator, aliasing excluded, detectors that fire on injected buzz, smear, lag and machine
pitch, a loop that escalates and falls back without ever returning a non-finite sample). None of it has
been trained at size, and "bulletproof" is a claim only a listening test on a trained model can make. What
can be said is that every named failure mode now has a mechanism against it, a detector for it, and a
fallback that is coherent by construction, which is more than a one-shot generator can say.

---

## 13. Hardening, round two (2.2): the production failure modes

Five more ways it fails in the real world were named, and each is right. The same discipline applies:
a mechanism against it, a detector for it, a fallback that is correct by construction.

### 13.1 Alignment drift on long tracks

**What is true.** A 5 ms error per chunk, accumulated across a five-minute track, is a quarter beat by the
last chorus; sub-frame F0 and a monotonic prior do not help if the clock itself drifts.

**What was built** (`helix_lm.music.longform`). Nothing accumulates: frame *i* is the samples
`[i*hop, (i+1)*hop)` of the whole track for the codec frames, the F0 curve, the durations and the output,
and every chunk is pasted back at its absolute position (`render_long`). Chunk edges snap to beats
(`plan_chunks`) and overlaps are crossfaded with raised cosines; the alignment prior is re-normalized
inside each chunk, so an error cannot be carried forward. The refiner's oscillator accumulates phase in
float64 and wraps it. Drift is **measured** (`drift_curve`: the sync lag in 20 s windows; `drift_rate`:
its slope in ms per minute and the constant offset) and a measured slope is cancelled by `time_warp` on
the controls before the affected chunks are regenerated. `Converter.convert` switches to this path above
30 s and reports the measured drift with every long conversion.

### 13.2 Extreme formant stretching

**What is true.** A tract does not scale with the pitch; a baritone pushed two octaves up through a
formant-invariant model is a hollow, resonance-less sound, because physics has nothing to offer there.

**What was built** (`helix_lm.music.register`). The reference voice's **tract scale** is measured from its
first three formants (LPC); its **comfortable range** from its F0 distribution. `register_plan` says how far
the melody leaves that range, which register the voice will be in there (chest, head, falsetto, whistle,
beyond physics), what the plan should do (sing as written; F1 tunes to F0 on the top notes, Joliveau et
al. 2004; move the top line down an octave; compress), and the **bounded formant ratio** the model may
apply: a quarter of the pitch excursion in log terms, capped at the 15-20 % a real tract spans. The model
warps its sketched envelope by that ratio (`warp_envelope` inside `TriStreamForSinging.sketch`), the swap
pipeline logs the register verdict, and `hollow_score` detects the failure when a conversion was pushed
past the limit anyway (F1-region deficit against the reference, flat formant contrast). Two octaves up from
a baritone is reported as "beyond physics" with the melody moved; it is not pretended.

### 13.3 Air above the codec's Nyquist

**What is true.** A 24 kHz pipeline has nothing above 12 kHz, whatever the frame rate; the air of a
sibilant and the breath of a close microphone are there, and a studio monitor shows their absence.

**What was built** (`helix_lm.audio.refiner.AirBandExtension`). The body is taken from 24 to 48 kHz and the
top band is filled with shaped noise whose per-band gains are predicted from the body's spectrum and
whose time envelope follows the 4-11 kHz band (sibilance drives air): spectral band replication with the
gains learned from 48 kHz recordings instead of transmitted. `air_report` measures the missing air in dB
against a reference and `pre_echo_score` the backward smear of transients that is the "mp3" sound; both
join the artifact report at 44.1 kHz and above. For conversion the source vocal's own air is transplanted
where it exists (`transient_transplant` with a higher cutoff), because unvoiced air carries almost no identity.

### 13.4 Out-of-distribution audio and separator residue

**What is true.** Development audio is dry stems; user audio is rooms, noise, rips, clipping and
phase-cancelled separator output, and a refiner will sharpen the damage with the voice.

**What was built** (`helix_lm.music.intake`). `assess_input` measures bandwidth, noise floor, clipping,
room T60, codec holes and summarizes them as a quality in [0, 1]; `clean_input` runs only the steps that
measurement calls for (declipping, click repair, spectral-gate denoising from the signal's own quiet
frames, Lebart late-reverb suppression) and records each; `separation_sanity` catches phase-cancelled
residue (negative vocal/instrumental correlation in the voice band) and invented energy. The quality
travels with the batch as a **condition** (`quality_in` in the filter stream), so the model is told the
input is dirty rather than left to imitate the dirt, and it scales the refiner's residual path (`trust`),
so a rough input gets the physical paths and the codec but not an amplifier. Training sees the same
world: `augment` now adds codec holes, a noise floor and anti-phase separator residue to its rooms,
echoes, muffles and loud instrumentals.

### 13.5 Zero-shot reference contamination

**What is true.** A ten-second reference with a chorus behind it, a bathroom around it or a shout in it
becomes, through a pooled embedding, the singer's "permanent" timbre.

**What was built** (`intake.reference_embedding`). The reference is cleaned, cut into one-second windows,
and windows that are whispered or silent (little voicing), shouted (loud and bright well above the clip's
own norm) or too quiet are dropped; the rest are embedded one by one and combined by a **trimmed mean**
around the medoid, so one window cannot steer the voice. The report lists what was dropped and the
**contamination**: how far the naive pooled embedding sat from the robust one; the swap pipeline warns
when it is large. `condition_invariance_loss` trains the speaker encoder so that a clip and its augmented
twin (room, noise, music, band limit) embed the same and different voices do not. `Converter.embed` uses
the robust path by default.

### 13.6 Still not claimed

Every mechanism above runs and is tested at toy scale on a CPU (`tests/test_hardening2.py`). The
formant estimates are the phonetics recipe, good to a few percent on clean vowels and worse on dense
mixes; the intake's cleaners are classical and will not rescue a 32 kbps rip; the air band needs 48 kHz
training data you supply; and the drift controller corrects a measured slope, not an unmeasurable one.
What is claimed is narrower and true: each named failure now has a measurement, a mechanism and a
fallback, and the system reports which of them it used.

---

## 14. Hardening, round three (2.3): the traps in training

### 14.1 The melisma trap

A spoken syllable is short; a sung vowel can last four seconds under a run with vibrato. A rigid monotonic
prior would hurry it. The prior is now **sustain-aware**: `sustain_profile` reads the source stream and
measures how long the current pitch has been held (voiced, under half a semitone of motion per frame);
the alignment prior's width grows with it, up to four times (`sustain_sigma_gain = 3`), so a frame deep in
a held vowel may attend to that vowel's phoneme however far its centre is, while moving syllables keep the
narrow prior. Durations come from the data in training (melismas included) and from the duration head at
inference, fitted to the clip. The spoken domain also carries half the loss weight, so the duration head is
not a speech-pacing head.

### 14.2 Codebook starvation on the high register

A quantizer trained mostly on speech spends its codes between 300 Hz and 4 kHz. Three measures: clips
whose F0 reaches 500 Hz get a **fixed share of every codec batch** (`--high-register-share 0.25`); a
**high-band loss** above 6 kHz (weight 2) joins the codec's losses; and the **codebook audit**
(`codebook_usage_by_register`) runs every 2,000 steps on a fixed high-register and a fixed low-register
audit set, logs the share of codes each tier uses on each, writes `codec/audit.json` (pushed with the
checkpoints) and flags starvation when any upper tier uses under half as many codes on the high register
as on the low. If the flag shows, raise the share or the loss weight; the number is the evidence, not the hope.

### 14.3 The noise-floor shortcut

The singing sets are dry studio stems; Emilia is the wild. Any discriminator or encoder could learn
"clean means sung". So no domain is clean: every training loop (codec, refiner, TriStream) passes a clip
through `augment` with probability `--augment-prob 0.5`, which now includes a **per-band synthetic room**
(`synthetic_room`: random decays with highs dying faster, pre-delay, wet mix) and a **random EQ**
(`random_eq`: tilt up to 6 dB per octave and one peaking band) besides the echoes, muffles, noise floors,
codec holes, loud instrumentals and separator residue it already had. For TriStream the augmented copy is
the **input** and the dry take stays the **target** (`batch["target_frames"]`), so the model learns to
listen through a room, never to paint one.

### 14.4 The formant wall

A baritone sent up to a falsetto hook crosses a register boundary, and a formant move taken as a step at
that boundary is a click. The formant move is now a **per-frame curve** (`tract_ratio_curve`): nothing
inside the voice's own range, a sigmoid ramp over three semitones past its top, a quarter of the excess in
log terms, capped at 1.2, held through unvoiced gaps and low-passed over 150 ms; `warp_envelope` applies it
frame by frame, and the converter carries it through long-form chunks. In training a **formant smoothness
loss** (weight 0.02) penalizes envelope jumps between frames of a held pitch. `register_break_report`
checks every boundary crossing of an output for a level jump over 12 dB or a click within 100 ms and the
molab `eval` reports `register_pops_per_minute`.

### 14.5 Index staleness at the drop

The one-block look-ahead (section 12.3) is always on. On top of it, the song-event track now switches on a
**two-block look-ahead** (`ahead2`) in the frames leading up to a downbeat or a section change: the trunk's
input carries the source and noise streams from two blocks ahead exactly where a drop, a key change or an
ad-lib on the one is coming, so the index decision at the boundary is made with the drop in hand. The local
strand covers the inside of the blocks as before. In long-form conversion the chunk edges snap to the
song's transitions (`transition_frames`: section starts and downbeats with a loudness surge), so the first
block after a transition is computed from the transition itself, never from the intro's context.

### 14.6 Measured here

Each mechanism is tested in `tests/test_hardening3.py`: the prior widens on held notes and not on moving
lines; the codebook audit returns per-tier usage and a starvation flag; the augmentation emits rooms and EQ
and the model trains with a dirty input against a dry target; the formant curve has no step larger than
0.02 and the pop detector catches a click at the crossing; the transition anchors land on chunk edges and
the extra look-ahead changes only the frames before the drop.
