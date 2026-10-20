# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
Audio for Anansi: raw waveforms in, raw waveforms out, with the detail kept.

* :mod:`frontend` -- a causal convolutional encoder that reads the waveform itself (no spectrogram) into
  25 frames a second, and the mirror decoder that writes one back;
* :mod:`codec` -- the split residual vector quantizer (one semantic codebook, seven acoustic) and the losses
  (multi-scale spectral, time-domain, commitment, adversarial) that train the pair into a codec; the discrete
  tokens are what the language model reads and writes, the continuous frames are what it listens with;
* :mod:`speaker` -- a small speaker encoder for voice embeddings, used beside in-context prompting for
  zero-shot voice cloning;
* :mod:`singing` -- pitch, note and lyric streams for singing voice synthesis;
* :mod:`omni` -- the multimodal wrapper: text tokens and audio frames interleaved in one Anansi sequence,
  audio emitted as codec tokens with Moshi's acoustic delay and a depth transformer for the residual codebooks;
* :mod:`data` -- the catalogue of speech, singing and general-audio datasets, and the waveform reader.

Every piece runs on a CPU at toy scale and is tested that way (``tests/test_audio.py``). None has been trained
at scale here: ``docs/ANANSI.md`` lists the published numbers that set the expectations.
"""

from .codec import AudioCodec, CodecConfig, ResidualVQ, codec_losses
from .frontend import WaveDecoder, WaveEncoder
from .omni import OmniConfig, OmniForCausalLM, apply_delay, undo_delay
from .singing import SingingConditioner, hz_to_bins
from .speaker import SpeakerEncoder

__all__ = [
    "AudioCodec",
    "CodecConfig",
    "OmniConfig",
    "OmniForCausalLM",
    "ResidualVQ",
    "SingingConditioner",
    "SpeakerEncoder",
    "WaveDecoder",
    "WaveEncoder",
    "apply_delay",
    "codec_losses",
    "hz_to_bins",
    "undo_delay",
]
