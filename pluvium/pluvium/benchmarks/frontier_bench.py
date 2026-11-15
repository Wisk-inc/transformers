# Copyright 2026 Nathan. Apache-2.0.
"""
The frontier benchmark from start to finish, one place at a time.

For each place:

1. **store**: every component is fitted at every quarterly refit and its forecasts are stored
   (:func:`pluvium.frontier.store.build_store`). This is the slow part (about an hour per place on four cores) and it
   resumes where it stopped;
2. **evaluate**: the declared design and its ablations are scored on the store's test year
   (:func:`pluvium.frontier.evaluate.evaluate`);
3. **tail** and **scenarios**: the heavy end and the coupled scenarios are scored from the chances kept in step 2
   (:mod:`.tail_bench`, :mod:`.scenario_bench`).

The designs are fixed here, in :data:`DESIGNS`. ``stack`` is the design that carries the name Pluvium Blend.
``declared`` is the design exactly as it was written down, with its settings, before the test year was scored: the
same, with the record-storm hand-over always on. The first places scored showed that the hand-over hurts in some
places and helps in others, so each place now decides from its own training year: the hand-over is used only where
it improved the forecast of storms withheld from training (:func:`staged_verdict`,
:mod:`pluvium.tails.withheld`). Both designs are scored everywhere, so the difference is on the page. The
remaining designs take parts away so that each part's worth can be read off.

    python -m pluvium.benchmarks.frontier_bench --archive ARCHIVE --truth TRUTH --out OUT --places alps,caribbean
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..frontier.evaluate import PoolDesign, evaluate
from ..frontier.store import FrontierConfig, Store, build_store
from ..util import log
from . import scenario_bench, tail_bench

POOL = dict(bands=(0, 5, 11), smooth=0.0)
#: the hand-over's declared setting: where it starts (a quantile of the training forecasts) and how fast it rises
POOL_RECORD = (0.999, 0.6931)
#: Pluvium Blend first, then the design as first declared, then the same with parts taken away
DESIGNS = (PoolDesign(name="stack", online=True, eta=1.0, record=True, staged=True, **POOL),
           PoolDesign(name="declared", online=True, eta=1.0, record=True, **POOL),
           PoolDesign(name="pool", **POOL),
           PoolDesign(name="pool_online", online=True, eta=1.0, **POOL),
           PoolDesign(name="pool_record", record=True, **POOL))
PLACES = ("caribbean", "maritime", "alps", "pacific_nw", "great_plains", "andes_south")


def staged_verdict(cfg: FrontierConfig, place: str, country: dict | None = None, top: int = 4) -> dict:
    """The place's verdict on the hand-over, from storms staged inside its training year: the four wettest days are
    withheld from training and then forecast with and without it (:mod:`pluvium.tails.withheld`). The experiment's
    full result is kept beside the store as ``staged.json``; the verdict is recomputed from it."""
    from dataclasses import replace

    from ..frontier.store import place_table
    from ..sources.archive import ARCHIVE_SYSTEMS
    from ..tails import withheld

    path = Path(cfg.out) / "store" / place / "staged.json"
    if path.exists():
        return withheld.verdict(json.loads(path.read_text()))
    cells, table, _ = place_table(replace(cfg, test_end=cfg.test_start), place, country)        # the training year only
    result = withheld.withheld_experiment(table, len(cells), list(ARCHIVE_SYSTEMS), top=top, quantiles=(POOL_RECORD[0],),
                                          widths=(POOL_RECORD[1],), log=log)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=1, default=float))
    return withheld.verdict(result)


def evaluate_place(store_folder, out, designs=DESIGNS, case_days=None, archive=None, region: str | None = None,
                   scenario_every: int = 3, verdict: dict | None = None) -> dict:
    """Score a place's store and write ``{place}.json``, ``{place}_keep.npz``, ``{place}_tail.json`` and, when the
    forecast ``archive`` is given, ``{place}_scenarios.json`` into ``out``. Steps whose file exists are skipped.
    ``verdict`` is the place's :func:`staged_verdict`; without one the design that follows it hands over."""
    store = Store(store_folder)
    place = store.folder.name
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    days = tuple(store.config.get("case_days", ())) if case_days is None else tuple(case_days)
    result_path, keep_path = out / f"{place}.json", out / f"{place}_keep.npz"
    if result_path.exists() and keep_path.exists():
        result = json.loads(result_path.read_text())
    else:
        keep = {}
        result = evaluate(store, list(designs), case_days=days, keep=keep, verdict=verdict)
        blocks = np.array(sorted(keep))
        np.savez_compressed(keep_path, blocks=blocks, p=np.stack([keep[int(b)] for b in blocks]))
        result_path.write_text(json.dumps(result, default=float))
        log(f"frontier benchmark: {place}: scored")
    d = np.load(keep_path)
    tail_path = out / f"{place}_tail.json"
    if not tail_path.exists():
        tail_path.write_text(json.dumps(tail_bench.run(store, d["blocks"], d["p"], case_days=days), indent=1))
    scen_path = out / f"{place}_scenarios.json"
    if archive is not None and not scen_path.exists():
        from ..sources.archive import ArchiveReader

        res = scenario_bench.run(store, d["blocks"], d["p"], ArchiveReader(archive), region or place, every=scenario_every)
        scen_path.write_text(json.dumps(res, indent=1))
    return result


def run(cfg: FrontierConfig, places=PLACES, countries: dict | None = None) -> dict:
    """Build the stores and score them. ``countries`` maps a place to the country settings
    :func:`pluvium.frontier.store.build_store` takes (a country at Pluvium's own resolution, with its network)."""
    results = {}
    out = Path(cfg.out) / "results"
    for place in places:
        country = (countries or {}).get(place)
        folder = build_store(cfg, place, country)
        region = country.get("region", place) if country else place
        results[place] = evaluate_place(folder, out, archive=cfg.archive, region=region,
                                        verdict=staged_verdict(cfg, place, country))
    return results


def main(argv=None) -> None:
    import argparse

    p = argparse.ArgumentParser(prog="pluvium frontier-bench", description="The frontier benchmark, place by place.")
    p.add_argument("--archive", required=True, help="the staged forecast archive (pluvium.sources.archive)")
    p.add_argument("--truth", required=True, help="the staged daily CMORPH truth (pluvium.sources.cmorph_daily)")
    p.add_argument("--out", required=True, help="stores go in OUT/store, results in OUT/results")
    p.add_argument("--places", default=",".join(PLACES))
    a = p.parse_args(argv)
    cfg = FrontierConfig(archive=a.archive, truth=a.truth, out=a.out)
    run(cfg, [x.strip() for x in a.places.split(",") if x.strip()])


if __name__ == "__main__":
    main()
