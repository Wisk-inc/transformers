# Copyright 2026 Nathan. Apache-2.0.
"""
The AI weather models behind two of the six systems Pluvium reads, and where their weights are published.

Pluvium does not run these models. It reads the forecasts NOAA and ECMWF make with them. The weights are listed here
so that a copy can be kept beside Pluvium (:mod:`pluvium.upstream.mirror`): a forecast feed can be renamed or
switched off, as NOAA's GraphCastGFS feed was in May 2026, and the weights are what would be needed to make those
forecasts again.

Every address, file name and licence below was read from the publisher's own pages on 15 November 2026:

* DeepMind's bucket holds a ``LICENSE`` file with the text of CC BY 4.0, and the readme of
  ``google-deepmind/graphcast`` says the code is Apache 2.0 and "all other materials" are CC BY 4.0.
* NOAA's entry in the Registry of Open Data on AWS says its GraphCast GFS products are released under CC0 and
  points to DeepMind's licence for GraphCast itself. The weights NOAA publishes were fine-tuned from DeepMind's.
* ECMWF's model cards say the inference weights are CC BY 4.0.

Licences change. :func:`pluvium.upstream.fetch.fetch_model` keeps the publisher's own licence file and model card
with every copy, so what was true on the day of the copy travels with it.
"""

from __future__ import annotations

from dataclasses import dataclass

DEEPMIND = "https://storage.googleapis.com/dm_graphcast/"
NOAA = "https://noaa-nws-graphcastgfs-pds.s3.amazonaws.com/graphcastgfs_weights/"
CC_BY = ("CC BY 4.0", "https://creativecommons.org/licenses/by/4.0/", "CC-BY-4.0.txt")
CC0 = ("CC0 1.0", "https://creativecommons.org/publicdomain/zero/1.0/", "CC0-1.0.txt")
GRAPHCAST_PAPER = ("Lam and others, Learning skillful medium-range global weather forecasting, Science 382, "
                   "1416-1421 (2023), doi:10.1126/science.adi2336")
AIFS_PAPER = "Lang and others, AIFS: ECMWF's data-driven forecasting system, arXiv:2406.01465 (2024)"


@dataclass(frozen=True)
class Upstream:
    """One published set of weights.

    ``kind`` is ``"http"`` for files in a public bucket (``files`` are paths under ``location``, or pairs of a path
    and the name to save it under) or ``"hub"`` for a model repository on the Hugging Face Hub (``location`` is its
    id and every file in it is copied).
    ``feeds`` names the Pluvium input the model stands behind: ``"graphcast"`` or ``"aifs"``.
    """

    name: str
    title: str
    producer: str
    feeds: str
    kind: str
    location: str
    files: tuple
    licences: tuple            # (short name, address, file name in pluvium/upstream/licences), one per licence
    attribution: str
    homepage: str
    papers: tuple
    note: str
    size_mb: int
    current: bool = False      # makes the forecasts Pluvium reads today

    @property
    def licence(self) -> str:
        return " and ".join(name for name, _, _ in self.licences)


MODELS = (
    Upstream(
        name="graphcast",
        title="GraphCast (Google DeepMind)",
        producer="Google DeepMind",
        feeds="graphcast",
        kind="http",
        location=DEEPMIND,
        files=(
            "graphcast/params/GraphCast_operational - ERA5-HRES 1979-2021 - resolution 0.25 - pressure levels 13 - "
            "mesh 2to6 - precipitation output only.npz",
            "graphcast/params/GraphCast - ERA5 1979-2017 - resolution 0.25 - pressure levels 37 - mesh 2to6 - "
            "precipitation input and output.npz",
            "graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - "
            "precipitation input and output.npz",
            "graphcast/stats/diffs_stddev_by_level.nc",
            "graphcast/stats/mean_by_level.nc",
            "graphcast/stats/stddev_by_level.nc",
            ("LICENSE", "LICENSE.deepmind.txt"),               # the bucket's own licence file, kept under this name
        ),
        licences=(CC_BY,),
        attribution="GraphCast weights and statistics, Copyright Google LLC, licensed under CC BY 4.0. Copied "
                    "unchanged from gs://dm_graphcast. " + GRAPHCAST_PAPER + ".",
        homepage="https://github.com/google-deepmind/graphcast",
        papers=("arXiv:2212.12794", "doi:10.1126/science.adi2336"),
        note="The three checkpoints DeepMind released, with the normalisation statistics they need. NOAA started "
             "from the 13-level one.",
        size_mb=434,
    ),
    Upstream(
        name="noaa-graphcastgfs",
        title="GraphCastGFS and AIGFS (NOAA)",
        producer="NOAA National Centers for Environmental Prediction",
        feeds="graphcast",
        kind="http",
        location=NOAA,
        files=(
            "params/AIGFSv1.1.npz",
            "params/GCGFSv2_finetuned - GDAS - ERA5 - resolution 0.25 - pressure levels 13 - mesh 2to6 - "
            "precipitation output only.npz",
            "params/params_gdas-gdas_epoch_30.npz",
            "stats/diffs_stddev_by_level.nc",
            "stats/mean_by_level.nc",
            "stats/stddev_by_level.nc",
        ),
        licences=(CC0, CC_BY),
        attribution="Weights and statistics published by NOAA in s3://noaa-nws-graphcastgfs-pds, which NOAA releases "
                    "under CC0. They were fine-tuned from Google DeepMind's GraphCast, whose weights are licensed "
                    "under CC BY 4.0, so DeepMind's attribution is kept with them. " + GRAPHCAST_PAPER + ".",
        homepage="https://registry.opendata.aws/noaa-nws-graphcastgfs-pds/",
        papers=("arXiv:2212.12794",),
        note="NOAA's own weights for the forecasts Pluvium reads as its GraphCast input: GraphCast fine-tuned on "
             "NOAA's GDAS analyses (GraphCastGFS, read until 5 May 2026) and its successor AIGFS (read since).",
        size_mb=433,
        current=True,
    ),
    Upstream(
        name="ecmwf-aifs-single-2.0",
        title="AIFS Single 2.0 (ECMWF)",
        producer="European Centre for Medium-Range Weather Forecasts",
        feeds="aifs",
        kind="hub",
        location="ecmwf/aifs-single-2.0",
        files=(),
        licences=(CC_BY,),
        attribution="AIFS Single v2.0 inference weights, Copyright ECMWF, licensed under CC BY 4.0. Copied unchanged "
                    "from huggingface.co/ecmwf/aifs-single-2.0. " + AIFS_PAPER + ".",
        homepage="https://huggingface.co/ecmwf/aifs-single-2.0",
        papers=("arXiv:2406.01465", "arXiv:2509.18994", "arXiv:2604.25559"),
        note="ECMWF's operational AI model since 12 May 2026: the forecasts Pluvium reads as its AIFS input today.",
        size_mb=998,
        current=True,
    ),
    Upstream(
        name="ecmwf-aifs-single-1.1",
        title="AIFS Single 1.1 (ECMWF)",
        producer="European Centre for Medium-Range Weather Forecasts",
        feeds="aifs",
        kind="hub",
        location="ecmwf/aifs-single-1.1",
        files=(),
        licences=(CC_BY,),
        attribution="AIFS Single v1.1 inference weights, Copyright ECMWF, licensed under CC BY 4.0. Copied unchanged "
                    "from huggingface.co/ecmwf/aifs-single-1.1. " + AIFS_PAPER + ".",
        homepage="https://huggingface.co/ecmwf/aifs-single-1.1",
        papers=("arXiv:2406.01465",),
        note="The version ECMWF ran before 12 May 2026. It replaced 1.0 to remove spurious rain points.",
        size_mb=996,
    ),
    Upstream(
        name="ecmwf-aifs-single-1.0",
        title="AIFS Single 1.0 (ECMWF)",
        producer="European Centre for Medium-Range Weather Forecasts",
        feeds="aifs",
        kind="hub",
        location="ecmwf/aifs-single-1.0",
        files=(),
        licences=(CC_BY,),
        attribution="AIFS Single v1.0 inference weights, Copyright ECMWF, licensed under CC BY 4.0. Copied unchanged "
                    "from huggingface.co/ecmwf/aifs-single-1.0. " + AIFS_PAPER + ".",
        homepage="https://huggingface.co/ecmwf/aifs-single-1.0",
        papers=("arXiv:2406.01465",),
        note="The first AIFS version ECMWF supported operationally. It replaced the experimental 0.2.1.",
        size_mb=996,
    ),
    Upstream(
        name="ecmwf-aifs-single-0.2.1",
        title="AIFS Single 0.2.1 (ECMWF)",
        producer="European Centre for Medium-Range Weather Forecasts",
        feeds="aifs",
        kind="hub",
        location="ecmwf/aifs-single-0.2.1",
        files=(),
        licences=(CC_BY,),
        attribution="AIFS Single v0.2.1 inference weights, Copyright ECMWF, licensed under CC BY 4.0. Copied "
                    "unchanged from huggingface.co/ecmwf/aifs-single-0.2.1. " + AIFS_PAPER + ".",
        homepage="https://huggingface.co/ecmwf/aifs-single-0.2.1",
        papers=("arXiv:2406.01465",),
        note="The experimental AIFS that ECMWF published forecasts from before version 1.0.",
        size_mb=1011,
    ),
)


def get(name: str) -> Upstream:
    """The entry called ``name``."""
    for model in MODELS:
        if model.name == name:
            return model
    raise KeyError(f"no upstream model called {name!r}; known: {', '.join(m.name for m in MODELS)}")


def select(names=None, current_only: bool = False) -> list:
    """The entries asked for, in the registry's order (all of them when ``names`` is empty)."""
    wanted = [n for n in (names or []) if n]
    for name in wanted:
        get(name)                                              # a wrong name is an error, not a silent skip
    return [m for m in MODELS if (not wanted or m.name in wanted) and (m.current or not current_only)]


__all__ = ["MODELS", "Upstream", "get", "select"]
