# Copyright 2026 Nathan. Apache-2.0.
"""
From nothing to a forecast in one call.

``pluvium now --country JM`` (or :func:`forecast` in Python) does whatever is still missing, in this order:

1. sets the country up: borders, terrain, every town (seconds);
2. gets a trained Pluvium Blend for it. If one is published in the Hub repository (``blends/<area>/``) it is
   downloaded, a few hundred kilobytes. If not, one is trained on the most recent year of archived runs, which takes
   about an hour and a half, nearly all of it downloads;
3. reads the newest run of each of the six systems for the country (a few megabytes) and writes the 14-day table.

The second call finds everything in place and takes about a minute. Nobody has to fetch or convert the six systems'
output by hand: Pluvium asks each public archive for the rain fields over the country and nothing else.

``pluvium blend publish`` uploads a trained blend so that step 2 is a download for everybody else.
A downloaded blend is opened in a way that cannot run code (:func:`pluvium.blendlive.read_state`).
"""

from __future__ import annotations

import json
import shutil
import tempfile
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .util import hour_of, hour_to_datetime64, hub_quiet, hub_refusal, log

DEFAULT_REPO = "CorxLabs/Pluvium-500"
HUB = "https://huggingface.co"
FOLDER = "blends"
#: when in the day it usually rains over the area, by month (:func:`add_timing`)
TIMING = "timing.npz"
#: what a published blend is made of. ``returns.npz`` is missing from blends trained before 0.3, and ``timing.npz``
#: from those trained before 0.5 (without it the forecast has no time of day)
BLEND_FILES = ("blend.json", "blend.pt", "climatology.npz", "returns.npz", TIMING)
REQUIRED = BLEND_FILES[:3]
#: when NOAA's bucket cannot be asked how far the satellite record goes: assume it ends this many days ago
FALLBACK_LAG_DAYS = 200


def codes_of(country) -> list:
    """``"do,ht"`` or ``["DO", "HT"]`` -> ``["DO", "HT"]``."""
    parts = country.split(",") if isinstance(country, str) else list(country)
    codes = [c.strip().upper() for c in parts if c.strip()]
    if not codes:
        raise ValueError("no country given: use ISO codes such as JM or DO,HT")
    return codes


def default_root(codes) -> Path:
    """Where a country's files live when no folder is named: ``~/pluvium-JM``."""
    return Path.home() / ("pluvium-" + "-".join(codes))


def recent_year(now: int | None = None, truth_last: int | None = None) -> tuple:
    """The most recent full year a blend can be trained on, as ``(start, end)``.

    A blend learns from the rain that fell after each forecast, so the year must end 15 days before the last day
    the satellite record holds (``truth_last``, hours since 1970). NOAA publishes that record about five months
    late; when ``truth_last`` is not given it is read from NOAA's bucket, and when that cannot be reached the year
    ends ``FALLBACK_LAG_DAYS`` before ``now``."""
    now = int(datetime.now(timezone.utc).timestamp() // 3600) if now is None else int(now)
    if truth_last is None:
        from .sources.cmorph_daily import truth_until

        truth_last = truth_until(now)
    if truth_last is None:
        truth_last = now - now % 24 - 24 * (FALLBACK_LAG_DAYS - 15)
    end = truth_last - truth_last % 24 - 24 * 15
    return str(hour_to_datetime64(end - 24 * 365))[:13], str(hour_to_datetime64(end))[:13]


def catch_up(model, root, now: int | None = None, max_days: int = 14, say=log) -> list:
    """Update a blend's online weights from the days whose rain has been published since it last learned, at most
    ``max_days`` of them, and save it. Returns the days learned from.

    The satellite record arrives months late, so on most days there is nothing new and this says so. When NOAA
    publishes another month, each call works through two weeks of it."""
    from .sources.cmorph_daily import stage_daily, truth_until

    hour = int(datetime.now(timezone.utc).timestamp() // 3600) if now is None else int(now)
    date = lambda h: str(hour_to_datetime64(int(h)))[:10]      # noqa: E731
    until, since = truth_until(hour), model.first_unlearned()
    if until is None:
        say("blend: NOAA's rain record could not be reached, so nothing was learned")
        return []
    if since > until:
        say(f"blend: nothing new to learn from. NOAA's satellite rain record ends on {date(until)} (it is published "
            f"about five months late) and this blend has learned up to it")
        return []
    stop = min(until, since + 24 * 3 * max_days)                # how far ahead one call looks for days with rain
    region = {model.area_json["name"]: tuple(model.area_json["target_box"])}
    truth = stage_daily(Path(root) / "truth", region, date(since) + "T00", date(stop + 24) + "T00")
    learned = model.learn(Path(root) / "live_archive", truth, hour, max_days=max_days, since=since, until=stop)
    if learned:
        model.save(Path(root) / "blend")
        behind = (until - hour_of(learned[-1] + "T00")) // 24
        say(f"blend: learned from {len(learned)} day(s), {learned[0]} to {learned[-1]}"
            + (f"; {behind} more day(s) are published and waiting for the next run" if behind > 0 else ""))
    else:
        say(f"blend: no day from {date(since)} to {date(stop)} has rain on record yet")
    return learned


def setup(codes, root, box=None, say=log):
    """Prepare a folder for some countries: the area, its terrain and provinces, and every town. Returns
    ``(area, geography, towns)``."""
    from .area import Area
    from .domain import Geography, build_gazetteer

    root = Path(root).expanduser()
    area = Area.for_countries(list(codes), box=box, cache=root / "cache" / "areas")
    root.mkdir(parents=True, exist_ok=True)
    area.save(root / "area.json")
    geo = Geography.build(cache=root / "cache" / "geography", area=area)
    geo.save(root / "geography")
    towns = build_gazetteer(geo.terrain, countries=area.iso2, cache=root / "cache" / "geonames", box=area.target_box)
    towns.points.save(root / "towns.npz")
    return area, geo, towns


# ------------------------------------------------------------------------------ blends on the Hub --

def blend_url(repo: str, name: str, file: str) -> str:
    return f"{HUB}/{repo}/resolve/main/{FOLDER}/{name}/{file}"


def _get(url: str, token: str | None = None) -> bytes | None:
    """The bytes at ``url``, or None when there is nothing there (or it may not be read)."""
    if not url.startswith("http"):                              # a local folder stands in for the Hub in tests
        path = Path(url[7:] if url.startswith("file://") else url)
        return path.read_bytes() if path.exists() else None
    headers = {"User-Agent": "pluvium-quick/1"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120.0) as response:
            return response.read()
    except urllib.error.HTTPError as error:
        if error.code in (401, 403, 404):
            return None
        raise


def same_area(a: dict, b: dict) -> bool:
    """Whether two saved areas are the same place: the same name and the same box, to a hundredth of a degree."""
    return a.get("name") == b.get("name") and np.allclose(a.get("target_box", [0] * 4), b.get("target_box", [1] * 4),
                                                           atol=0.01)


def fetch_blend(repo: str, area_json: dict, root, token: str | None = None, say=log, url=None) -> bool:
    """Download the blend published for this area into ``root/blend``. True when there was one, it is for this
    area, and it opens safely; otherwise nothing is left in ``root/blend`` and False is returned. ``url`` maps
    ``(repo, area name, file)`` to an address (default: :func:`blend_url`)."""
    import os

    url = url or blend_url
    token = token or os.environ.get("HF_TOKEN") or None        # only needed for a private repository
    name = area_json["name"]
    head = _get(url(repo, name, "blend.json"), token)
    if head is None:
        return False
    try:
        published = json.loads(head)
    except ValueError:
        return False
    if not same_area(published.get("area", {}), area_json):
        say(f"{repo} has a blend called {name}, but for a different box; not used")
        return False
    target = Path(root).expanduser() / "blend"
    stage = Path(tempfile.mkdtemp(prefix="pluvium-blend-"))
    try:
        (stage / "blend.json").write_bytes(head)
        for file in BLEND_FILES[1:]:
            body = _get(url(repo, name, file), token)
            if body is None:
                if file in REQUIRED:
                    say(f"{repo}: the blend for {name} is incomplete ({file} is missing); not used")
                    return False
                continue
            (stage / file).write_bytes(body)
        from .frontier.model import load_blend

        try:
            load_blend(stage, trusted=False)                    # opens without running code, or not at all
        except Exception as error:
            say(f"{repo}: the blend for {name} could not be opened safely ({type(error).__name__}: {error}); not used")
            return False
        target.mkdir(parents=True, exist_ok=True)
        for path in stage.iterdir():
            shutil.copy2(path, target / path.name)
        (target / "FETCHED").write_text(json.dumps({"repo": repo, "when": str(np.datetime64("now"))}))
        trained = published.get("meta", {}).get("trained", ["?", "?"])
        say(f"blend for {name}: downloaded from {repo} (trained on {trained[0][:10]} to {trained[1][:10]})")
        return True
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def _degrees(value: float, letters: str) -> str:
    """``17.40°N`` or ``78.67°W``: a latitude (``"NS"``) or a longitude (``"EW"``), with a letter in place of a sign."""
    return f"{abs(value):.2f}°{letters[0] if value >= 0 else letters[1]}"


def blend_card(model) -> str:
    """A short note uploaded beside a published blend: what it is for and how it was trained."""
    meta, area = model.meta, model.area_json
    s, n, w, e = area["target_box"]
    trained = meta.get("trained", ["?", "?"])
    lines = [f"# Pluvium Blend for {', '.join(area.get('countries', {}).values()) or area['name']}", "",
             f"This is the trained blend for the box from {_degrees(s, 'NS')} to {_degrees(n, 'NS')} and from "
             f"{_degrees(w, 'EW')} to {_degrees(e, 'EW')}. "
             f"`pluvium now --country {','.join(area.get('iso2', []))}` downloads it by itself and forecasts with it, "
             "you do not have to do anything with these files.", "",
             f"It was trained on {meta.get('runs', '?')} runs of each of the six forecasts, from {trained[0][:10]} to "
             f"{trained[1][:10]}, on {meta.get('cells', '?')} grid cells, {len(meta.get('components', []))} components pooled."]
    verdict = meta.get("handover")
    if verdict:
        staged = verdict.get("staged", {})
        how = "kept" if verdict.get("keep", True) else "switched off"
        storms, gain = staged.get("storms"), staged.get("gain")
        dated = isinstance(storms, (list, tuple))                # the days themselves, or only how many there were
        count = len(storms) if dated else storms
        why = ""
        if count and isinstance(gain, (int, float)) and np.isfinite(gain):
            days = f" ({', '.join(str(d) for d in storms)})" if dated else ""
            before, after = staged.get("calibrated"), staged.get("handed")
            scores = (f", {after:.3f} against {before:.3f}"
                      if all(isinstance(v, (int, float)) and np.isfinite(v) for v in (before, after)) else "")
            why = (f" At training the {count} wettest days{days} were held back and then forecast unseen, with the "
                   f"hand-over their Brier score at {staged.get('amount', 50):g} mm was {100 * abs(gain):.1f}% "
                   f"{'lower' if gain >= 0 else 'higher'}{scores}. Lower is better.")
        lines += ["", f"The hand-over of record storms to the raw systems is {how} here.{why}"]
    lines += ["", "The file `blend.pt` holds tensors and plain values only, Pluvium opens it without running any code "
              "in it.", ""]
    return "\n".join(lines)


def publish_blend(repo: str, root, token: str | None = None, private: bool = False, api=None, say=log) -> str:
    """Upload the blend trained in ``root`` to ``blends/<area>/`` of a Hub model repository, so that
    ``pluvium now`` can download it. Returns the address of the folder. Needs a write token (or ``HF_TOKEN``)."""
    import os

    from .frontier.model import load_blend

    root = Path(root).expanduser()
    model = load_blend(root / "blend")
    stage = Path(tempfile.mkdtemp(prefix="pluvium-publish-"))
    try:
        model.save(stage)                                       # written afresh, so always in the safe format
        load_blend(stage, trusted=False)
        if (root / "blend" / TIMING).exists():                  # when it usually rains there, if it was learned
            shutil.copy2(root / "blend" / TIMING, stage / TIMING)
        (stage / "README.md").write_text(blend_card(model), encoding="utf-8")
        if api is None:
            token = token or os.environ.get("HF_TOKEN")
            if not token:
                raise SystemExit("publishing needs a Hugging Face write token: set HF_TOKEN or pass one")
            from huggingface_hub import HfApi

            api = HfApi(token=token)
        name = model.area_json["name"]
        try:
            import warnings

            api.create_repo(repo, repo_type="model", exist_ok=True, private=private)
            # two things the Hub library says here that are not faults: the note beside the blend is not a model
            # card (it warns that it has no card metadata), and a blend published twice changes nothing
            with warnings.catch_warnings(), hub_quiet():
                warnings.simplefilter("ignore", UserWarning)
                api.upload_folder(folder_path=str(stage), path_in_repo=f"{FOLDER}/{name}", repo_id=repo,
                                  repo_type="model", commit_message=f"Pluvium Blend for {name}")
        except Exception as error:
            refusal = hub_refusal(error, repo)
            if refusal is None:
                raise
            raise refusal from None
        where = f"{HUB}/{repo}/tree/main/{FOLDER}/{name}"
        say(f"blend for {name}: published to {where}")
        return where
    finally:
        shutil.rmtree(stage, ignore_errors=True)


# ----------------------------------------------------------------------------------- the one call --

def ready(country, root=None, repo: str | None = DEFAULT_REPO, train: bool = True, start: str | None = None,
          end: str | None = None, handover: str = "auto", token: str | None = None, say=log) -> tuple:
    """Make sure a folder holds the area and a trained blend, doing only what is missing. Returns
    ``(root, model, how)`` where ``how`` is ``"local"``, ``"downloaded"`` or ``"trained"``."""
    from .area import Area
    from .frontier.model import FrontierBlend, load_blend

    codes = codes_of(country)
    root = default_root(codes) if root is None else Path(root).expanduser()
    if (root / "area.json").exists():
        area = Area.load(root / "area.json")
    else:
        say(f"setting up {', '.join(codes)} in {root} ...")
        area, _, _ = setup(codes, root, say=say)
    if (root / "blend" / "blend.json").exists():
        fetched = (root / "blend" / "FETCHED").exists()
        if fetched and repo and not (root / "blend" / TIMING).exists():   # downloaded before the repository had it
            top_up(repo, area.name, root, token)
        return root, load_blend(root / "blend", trusted=not fetched), "local"
    if repo and fetch_blend(repo, area.to_json(), root, token, say):
        return root, load_blend(root / "blend", trusted=False), "downloaded"
    if not train:
        raise SystemExit(f"no trained blend for {area.name} in {root} or in {repo}. Train one with: "
                         f"pluvium blend train --root {root}")
    if not (start and end):
        start, end = recent_year()
    say(f"no trained blend is published for {area.name}: training one on {start[:10]} to {end[:10]} "
        "(about an hour and a half the first time, nearly all of it downloads) ...")
    model = FrontierBlend.train(area, root, start=start, end=end, handover=handover)
    try:                                                        # the daily forecast stands without it
        add_timing(root, say=say)
    except Exception as error:
        say(f"the usual time of day of rain was not learned ({type(error).__name__}: {str(error)[:160]}); "
            f"`pluvium blend timing --root {root}` tries again")
    return root, model, "trained"


def add_timing(root, say=log) -> Path:
    """Learn when in the day it usually rains over the blend's area during its training period, and save it beside
    the blend as ``timing.npz``. The time-of-day forecast (:func:`periods_of`) needs it: checked for Jamaica, the
    systems' members alone place a wet day's rain worse than what is usual at the place. Reads the satellite
    record a second time, about a minute a month."""
    from . import timing as T
    from .frontier.model import load_blend

    root = Path(root).expanduser()
    model = load_blend(root / "blend", trusted=not (root / "blend" / "FETCHED").exists())
    trained = model.meta.get("trained")
    if not trained:
        raise ValueError("this blend does not say which period it was trained on")
    usual = T.fit_usual(root, model.area_json["name"], model.area_json["target_box"], trained[0], trained[1], say=say)
    path = root / "blend" / TIMING
    np.savez_compressed(path, lat=usual["lat"], lon=usual["lon"], given=usual["given"])
    return path


def top_up(repo: str, name: str, root, token: str | None = None, url=None) -> bool:
    """Fetch the usual time of day of rain (``timing.npz``) for a blend that was downloaded before its repository
    had it. True when it is now in the folder. Any failure leaves the folder as it was."""
    import os

    try:
        body = _get((url or blend_url)(repo, name, TIMING), token or os.environ.get("HF_TOKEN") or None)
        if body is None:
            return False
        import io

        with np.load(io.BytesIO(body), allow_pickle=False) as z:          # arrays only: nothing in it can run
            if not {"lat", "lon", "given"} <= set(z.files) or z["given"].shape[:2] != (12, 4):
                return False
        (Path(root).expanduser() / "blend" / TIMING).write_bytes(body)
        return True
    except Exception:
        return False


def places_of(root, model) -> tuple:
    """``(lat, lon, names, population)`` of the places to forecast: the country's towns, or the area's grid when
    there is no town list."""
    from .domain import Gazetteer, PointSet

    path = Path(root) / "towns.npz"
    if path.exists():
        towns = Gazetteer(PointSet.load(path))
        population = np.asarray(towns.points.extra.get("population", np.zeros(len(towns.points))), np.int64)
        return towns.points.lat, towns.points.lon, [str(x) for x in towns.names], population
    glat, glon = model.grid
    la, lo = np.meshgrid(glat, glon, indexing="ij")
    lat, lon = la.ravel(), lo.ravel()
    return lat, lon, [f"{a:.2f},{b:.2f}" for a, b in zip(lat, lon)], np.zeros(lat.size, np.int64)


def areas_of(root) -> list | None:
    """The first-level area each place is in (a parish in Jamaica, a province or state elsewhere), in the order of
    :func:`places_of`; None when the folder has no town list. The names are GeoNames', e.g. ``Saint Ann``."""
    from .domain import PointSet

    path = Path(root) / "towns.npz"
    if not path.exists():
        return None
    extra = PointSet.load(path).extra
    return [_tidy(str(x)) for x in extra["unit"]] if "unit" in extra else None


def _tidy(name: str) -> str:
    """One spelling for an area's name. GeoNames writes Jamaica's parishes three ways (``Portland Parish``,
    ``St. Elizabeth``, ``Clarendon``): the word Parish is dropped and St. is written out."""
    name = name.strip()
    if name.endswith(" Parish"):
        name = name[: -len(" Parish")]
    if name.startswith(("St. ", "St ")):
        name = "Saint " + name.split(" ", 1)[1]
    return name


def zone_of(root) -> str:
    """The time zone most of the area's places keep (an IANA name such as ``America/Jamaica``), or ``UTC`` when
    the folder has no town list."""
    from .domain import PointSet

    path = Path(root) / "towns.npz"
    if not path.exists():
        return "UTC"
    extra = PointSet.load(path).extra
    if "timezone" not in extra or not len(extra["timezone"]):
        return "UTC"
    zones, counts = np.unique(np.asarray(extra["timezone"]).astype(str), return_counts=True)
    return str(zones[np.argmax(counts)])


def usual_at(root, lat, lon, day0: int, days: int) -> np.ndarray | None:
    """How rain is usually spread over the day at the places, for each of ``days`` days from ``day0``:
    ``(days, 4, places)``, from ``blend/timing.npz`` (:func:`pluvium.timing.usual_timing` on the area's grid), or
    None when the blend has no such file."""
    path = Path(root) / "blend" / "timing.npz"
    if not path.exists():
        return None
    with np.load(path) as z:
        glat, glon, given = z["lat"], z["lon"], z["given"]                     # given: (12, 4, H, W)
    i = np.clip(np.round((np.asarray(lat) - glat[0]) / (glat[1] - glat[0])).astype(int), 0, glat.size - 1)
    j = np.clip(np.round((np.asarray(lon) - glon[0]) / (glon[1] - glon[0])).astype(int), 0, glon.size - 1)
    months = [int(str(hour_to_datetime64(int(day0) + 24 * d))[5:7]) for d in range(days)]
    return np.stack([given[m - 1][:, i, j] for m in months])


def periods_of(model, root, result: dict, lat, lon, days: int | None = None, issued: int | None = None) -> dict | None:
    """When in the day it rains, for the first days of a forecast (:mod:`pluvium.timing`). Returns None when the
    blend has no record of when it usually rains (``timing.npz``, :func:`add_timing`) or no system's six-hour
    fields can be read, otherwise

    * ``labels``  what each period is called in the area's clock, e.g. ``Sun 1 pm to 7 pm``;
    * ``starts``  the UTC hour each period begins, as text, and ``hours`` the same as hours since 1970;
    * ``chance``  ``(days, 4, places)`` the chance of at least 1 mm in the period;
    * ``systems`` the systems whose six-hour fields went in, ``zone`` the clock, ``days`` how many days are split;
    * ``issued``  the hour the forecast is made (now unless given): a period that ended before it is over.
    """
    from . import timing as T
    from .sources.archive import ArchiveReader

    days = min(days or T.DAYS, len(result["days"]))
    name = model.area_json["name"]
    day0 = hour_of(result["day0"] + "T00")
    usual = usual_at(root, lat, lon, day0, days)
    if usual is None:                                           # the members alone are not good enough to show
        return None
    runs = {system: hour_of(started) for system, started in result["runs"].items()}
    reader = ArchiveReader(Path(root) / "live_archive", {name: tuple(model.area_json["target_box"])})
    members = T.read_periods(reader, name, runs, day0, lat, lon, days)
    if not members:
        return None
    shares = T.member_shares(members, days)
    given = T.mixed(T.conditional(shares, usual), usual)
    daily = np.stack([np.asarray(result["days"][d]["p_ge_1"], np.float64) for d in range(days)])
    zone = zone_of(root)
    hours = [day0 + T.HOURS * k for k in range(T.PERIODS * days)]
    if issued is None:
        issued = int(datetime.now(timezone.utc).timestamp() // 3600)
    return {"labels": T.labels(day0, zone, days), "chance": T.chances(daily, given),
            "starts": [str(hour_to_datetime64(h))[:13] for h in hours], "hours": hours, "issued": int(issued),
            "systems": sorted(members), "zone": zone, "days": days}


def _over(periods: dict, k: int) -> bool:
    """Whether period ``k`` ended before the forecast was made. A day of the forecast is a UTC day, so in Jamaica its
    first period is the evening before (7 pm to 1 am), already over when the forecast is read in the morning."""
    from . import timing as T

    return "hours" in periods and periods["hours"][k] + T.HOURS <= periods.get("issued", -1)


def write_periods(periods: dict, names, path, areas=None) -> Path:
    """One row per place and period of the first days: when it starts (UTC), what it is called in the area's
    clock, and the chance of at least 1 mm in it."""
    import csv

    path = Path(path)
    where = [] if areas is None else ["area"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["place", *where, "starts_utc", "period", "p_ge_1mm"])
        for k, label in enumerate(periods["labels"]):
            d, j = divmod(k, 4)
            for n, name in enumerate(names):
                w.writerow([name, *([areas[n]] if areas is not None else []), periods["starts"][k], label,
                            f"{periods['chance'][d, j, n]:.2f}"])
    return path


def _when(periods: dict | None, d: int, index, least: float = 0.2) -> str:
    """``most likely Sun 1 pm to 7 pm (48%)`` for day ``d`` at the places ``index`` (their median), or nothing when
    the day has no periods or rain is unlikely in all of them. A period that is already over is never named: until
    0.5.7 a Jamaica forecast read on Tuesday morning could say Tuesday's rain was most likely "Mon 7 pm to 1 am"."""
    if periods is None or d >= periods["days"]:
        return ""
    chance = np.median(periods["chance"][d][:, index].reshape(4, -1), axis=1)
    chance = np.where([_over(periods, 4 * d + j) for j in range(4)], -1.0, chance)
    j = int(np.argmax(chance))
    if chance[j] < least:
        return ""
    return f"   most likely {periods['labels'][4 * d + j]} ({100 * chance[j]:.0f}%)"


def _plain(name: str) -> str:
    """A name as people type it, for matching: lower case, and ``St`` or ``St.`` read as ``Saint``."""
    words = name.strip().lower().replace(".", " ").split()
    return " ".join("saint" if w in ("st", "ste") else w for w in words)


def by_area(result: dict, areas: list, days: int = 5, area: str | None = None, periods: dict | None = None) -> str:
    """The next ``days`` days for every first-level area (parish, province, state), or for those whose name
    contains ``area``. Each line gives the typical place in the area (the median of its places) and, in brackets,
    the range across its places: an area is not one number, and its wettest corner is what a warning is about.
    With ``periods`` (:func:`periods_of`) the first days also say when in the day rain is most likely."""
    from .blendlive import ALERT_NAMES

    names = sorted({a for a in areas if a})
    if area:
        names = [a for a in names if _plain(area) in _plain(a)]
        if not names:
            return f"no area called {area!r}: this country has " + ", ".join(sorted({a for a in areas if a}))
    where = np.asarray(areas, dtype=object)
    lines = ["each line: the typical place in the area, and in brackets the range across its places"]
    for name in names:
        idx = np.flatnonzero(where == name)
        lines.append(f"{name} ({idx.size:,} places)")
        for number, d in enumerate(result["days"][:days]):
            date = np.datetime64(d["date"]).astype(datetime)
            rain, heavy, amount = (np.asarray(d[k])[idx] for k in ("p_ge_1", "p_ge_25", "mean_mm"))
            worst = int(np.asarray(d["alert"])[idx].max()) if "alert" in d and len(d["alert"]) else 0
            flag = f"   [{ALERT_NAMES[worst]} for part of it]" if worst else ""
            lines.append(f"  {date:%a %d %b}   rain {100 * np.median(rain):3.0f}% ({100 * rain.min():.0f} to "
                         f"{100 * rain.max():.0f})   25 mm or more {100 * np.median(heavy):3.0f}% (up to "
                         f"{100 * heavy.max():.0f})   expected {np.median(amount):5.1f} mm (up to {amount.max():.1f})"
                         + _when(periods, number, idx) + flag)
    return "\n".join(lines)


def brief(result: dict, names: list, population: np.ndarray, top: int = 5, days: int = 5, place: str | None = None,
          areas: list | None = None, area: str | None = None, periods: dict | None = None, when: bool = False) -> str:
    """A few lines a person can read: the next ``days`` days at the ``top`` largest places, or at the places whose
    name contains ``place``, or at the largest places of the areas whose name contains ``area`` (with ``areas``,
    each place's area, from :func:`areas_of`). With ``periods`` (:func:`periods_of`) the first days say when in
    the day rain is most likely, and with ``when`` each of their four periods gets a line of its own."""
    from .blendlive import ALERT_NAMES

    inside = np.ones(len(names), bool)
    if area:
        if areas is None:
            return "this folder has no town list, so its places have no areas"
        inside = np.array([_plain(area) in _plain(a) for a in areas])
        if not inside.any():
            return f"no area called {area!r}: this country has " + ", ".join(sorted({a for a in areas if a}))
    if place:
        q = place.strip().lower()
        idx = [k for k, n in enumerate(names) if q in n.lower() and inside[k]]
        idx = sorted(idx, key=lambda k: (names[k].lower() != q, -int(population[k])))[:top]
        if not idx:
            return f"no place called {place!r} in this area"
    else:
        order = np.argsort(-np.asarray(population), kind="stable")
        idx = [int(k) for k in order if inside[k]][:top]
    lines = []
    for k in idx:
        lines.append(names[k] + (f", {areas[k]}" if areas is not None and areas[k] else ""))
        for number, d in enumerate(result["days"][:days]):
            date = np.datetime64(d["date"]).astype(datetime)
            alert = ALERT_NAMES[int(d["alert"][k])] if "alert" in d else ""
            beyond = "beyond the training record" if "beyond_record" in d and d["beyond_record"][k] else ""
            flags = ", ".join(x for x in (alert, beyond) if x)
            lines.append(f"  {date:%a %d %b}   rain {100 * d['p_ge_1'][k]:3.0f}%   25 mm or more {100 * d['p_ge_25'][k]:3.0f}%"
                         f"   expected {d['mean_mm'][k]:5.1f} mm" + ("" if when else _when(periods, number, [k]))
                         + (f"   [{flags}]" if flags else ""))
            if when and periods is not None and number < periods["days"]:
                for j in range(4):
                    label = periods["labels"][4 * number + j]
                    lines.append(f"      {label:<22} over" if _over(periods, 4 * number + j) else
                                 f"      {label:<22} rain {100 * periods['chance'][number, j, k]:3.0f}%")
    return "\n".join(lines)


def forecast(country, root=None, repo: str | None = DEFAULT_REPO, time=None, learn: bool = False, train: bool = True,
             handover: str | None = None, token: str | None = None, say=log, when: bool = True) -> dict:
    """The 14-day rain forecast for every town of ``country`` (ISO codes), from nothing if need be.

    Returns ``{"root", "table", "result", "names", "population", "areas", "periods", "periods_table", "model",
    "how"}``: the folder, the CSV written, the forecast as :meth:`pluvium.blendlive.BlendModel.forecast` returns
    it, the places, each place's parish or province (None without a town list), when in the day it rains on the
    first three days (:func:`periods_of`; None with ``when=False`` or when no six-hour fields can be read) with its
    own CSV, the model and how it was
    obtained (``"local"``, ``"downloaded"`` or ``"trained"``). ``learn=True`` first updates the online weights from
    the days whose rain is now known (slow the first time: it reads two weeks of past runs). ``time`` forecasts as
    of another UTC hour, ``handover`` overrules the staged storms' verdict for this run.
    """
    from .blendlive import write_table
    from .frontier.model import FrontierBlend

    root, model, how = ready(country, root, repo, train, token=token, say=say)
    now = None if time is None else hour_of(time)
    if learn and isinstance(model, FrontierBlend):
        catch_up(model, root, now, say=say)
    if handover and isinstance(model, FrontierBlend):          # after saving: the overrule is for this run only
        model.set_handover(handover)
    lat, lon, names, population = places_of(root, model)
    areas = areas_of(root)
    result = model.forecast(lat, lon, root / "live_archive", now=now)
    table = write_table(result, names, root / f"blend_{result['day0']}.csv", areas=areas)
    periods, periods_table = None, None
    if when:
        try:                                                    # the daily forecast stands without it
            periods = periods_of(model, root, result, lat, lon, issued=now)
        except Exception as error:
            say(f"the time of day was not worked out ({type(error).__name__}: {str(error)[:160]})")
        if periods is not None:
            periods_table = write_periods(periods, names, root / f"blend_{result['day0']}_periods.csv", areas=areas)
    return {"root": root, "table": table, "result": result, "names": names, "population": population, "areas": areas,
            "periods": periods, "periods_table": periods_table, "model": model, "how": how}


__all__ = ["DEFAULT_REPO", "areas_of", "blend_card", "blend_url", "brief", "by_area", "catch_up", "codes_of",
           "default_root", "fetch_blend", "forecast", "periods_of", "places_of", "publish_blend", "ready",
           "recent_year", "same_area", "setup", "usual_at", "write_periods", "zone_of"]
