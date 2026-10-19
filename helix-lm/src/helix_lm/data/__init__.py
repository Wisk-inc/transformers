# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
The data that trains Helix: where it comes from, how it is prepared, and how it is read back.

* :mod:`catalog` -- every source, with its Hub id, columns, license and share of the mixture.
* :mod:`prepare` -- stream a source, filter it, tokenize it, write uint16 shards; resumable.
* :mod:`jamaica` -- the Jamaican Patois sources and the filter that finds Jamaica in the English web.
* :mod:`identity` -- who the model is: CorxLabs, Jamaica, and the facts it should know about itself.
* :mod:`loader` -- the weighted, deterministic, resumable reader the trainer draws batches from.
* :mod:`agent` -- coding-agent trajectories and instruction data turned into chat samples for SFT.
"""

from .catalog import PRETRAIN_SOURCES, SFT_SOURCES, Source, verify_sources
from .loader import MixtureLoader, TokenSource

__all__ = ["PRETRAIN_SOURCES", "SFT_SOURCES", "MixtureLoader", "Source", "TokenSource", "verify_sources"]
