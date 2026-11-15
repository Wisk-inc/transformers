# Copyright 2026 Nathan. Apache-2.0.
"""Small shared pieces: logging, HTTP with retries and byte ranges, parallel fetching, and the hour clock."""

from __future__ import annotations

import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

#: Every time in Pluvium is an integer hour since 1970-01-01 00:00 UTC. Hour ``h`` is the interval
#: ``[h, h + 1)``: rain "in hour h" fell between those two instants.
EPOCH = np.datetime64("1970-01-01T00", "h")


def log(*parts) -> None:
    print(time.strftime("[%H:%M:%S]"), *parts, flush=True)


def hour_of(when) -> int:
    """Integer hours since 1970 for a datetime, numpy datetime64, ISO string or int (returned unchanged)."""
    if isinstance(when, (int, np.integer)):
        return int(when)
    if isinstance(when, datetime):
        if when.tzinfo is not None:
            when = when.astimezone(timezone.utc).replace(tzinfo=None)
        when = np.datetime64(when, "h")
    return int((np.datetime64(when, "h") - EPOCH).astype(np.int64))


def hour_to_datetime64(hour: int) -> np.datetime64:
    return EPOCH + np.timedelta64(int(hour), "h")


def hour_to_iso(hour: int) -> str:
    return str(hour_to_datetime64(hour))[:13] + ":00Z"


def year_hours(year: int) -> tuple[int, int]:
    """``[first, last + 1)`` hours of a calendar year."""
    return hour_of(np.datetime64(f"{year}-01-01T00", "h")), hour_of(np.datetime64(f"{year + 1}-01-01T00", "h"))


def calendar(hours: np.ndarray) -> dict:
    """Year, month (1-12), day of year (1-366) and UTC hour of day for integer hours."""
    t = EPOCH + np.asarray(hours, dtype=np.int64).astype("timedelta64[h]")
    years = t.astype("datetime64[Y]").astype(int) + 1970
    months = t.astype("datetime64[M]").astype(int) % 12 + 1
    day = (t.astype("datetime64[D]") - t.astype("datetime64[Y]")).astype(int) + 1
    hod = (t - t.astype("datetime64[D]")).astype(int)
    return {"year": years, "month": months, "doy": day, "hour": hod}


class FetchError(RuntimeError):
    pass


class PluviumError(RuntimeError):
    """Something a person can act on (a period with no data, a missing folder). The command line prints its message
    and stops, without a traceback."""


def hub_refusal(error: Exception, repo: str) -> PluviumError | None:
    """If ``error`` is the Hugging Face Hub refusing a token, the same thing in plain words; otherwise None."""
    code = getattr(getattr(error, "response", None), "status_code", None)
    owner = repo.split("/")[0]
    if code == 401:
        return PluviumError("The Hugging Face Hub did not accept the token (HTTP 401). Check that it was pasted "
                            "whole and has not been revoked.")
    if code == 403:
        return PluviumError(f"The token is valid but may not write to {repo} (HTTP 403). It needs write access to "
                            f"{owner}. A fine-grained token only reaches the accounts it was given: on the Hub, under "
                            f"Settings, Access Tokens, edit the token and add {owner} with write access to its "
                            "repositories and collections, or make a token of type Write.")
    return None


def fetch(url: str, start: int | None = None, end: int | None = None, timeout: float = 120.0, attempts: int = 5,
          missing_ok: bool = False) -> bytes | None:
    """
    GET ``url`` (optionally bytes ``start``-``end`` inclusive) with retries and backoff. ``file://`` paths and plain
    local paths are read from disk, which is how the tests stand in for a bucket. With ``missing_ok`` a 404 returns
    ``None`` instead of raising (a gap in an archive is data, not an error).
    """
    if url.startswith("file://") or (not url.startswith("http") and os.path.exists(url)):
        path = url[7:] if url.startswith("file://") else url
        if not os.path.exists(path):
            if missing_ok:
                return None
            raise FetchError(f"no such file: {path}")
        with open(path, "rb") as handle:
            if start is not None:
                handle.seek(start)
                return handle.read(None if end is None else end - start + 1)
            return handle.read()
    headers = {"User-Agent": "pluvium/0.1"}
    if start is not None:
        headers["Range"] = f"bytes={start}-" + ("" if end is None else str(end))
    last = None
    for k in range(attempts):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            if error.code in (403, 404) and missing_ok:
                return None
            if error.code in (400, 401, 403, 404):
                raise FetchError(f"{url}: HTTP {error.code}") from error
            last = error
        except Exception as error:  # timeouts, resets, proxy hiccups: retry with backoff
            last = error
        time.sleep(min(30.0, 1.5 * 2**k))
    raise FetchError(f"{url}: {type(last).__name__}: {last}")


def parallel_map(function, items, workers: int = 16, progress: str | None = None, every: float = 30.0) -> list:
    """``[function(x) for x in items]`` on a thread pool (network-bound work), in order, with a progress line every
    ``every`` seconds when ``progress`` names the job."""
    items = list(items)
    if not items:
        return []
    out = [None] * len(items)
    started = last = time.time()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(function, item): i for i, item in enumerate(items)}
        done = 0
        from concurrent.futures import as_completed

        for future in as_completed(futures):
            out[futures[future]] = future.result()
            done += 1
            if progress and time.time() - last > every:
                rate = done / max(1e-6, time.time() - started)
                left = (len(items) - done) / max(rate, 1e-9) / 60
                log(f"  {progress}: {done:,} of {len(items):,} ({rate:.1f}/s, about {left:.0f} min left)")
                last = time.time()
    return out


def atomic_save(path: str | Path, writer) -> Path:
    """Write through a temporary sibling and rename, so an interrupted save never leaves a torn file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    writer(tmp)
    if not tmp.exists() and Path(str(tmp) + ".npy").exists():  # np.save appends .npy to a name without it
        tmp = Path(str(tmp) + ".npy")
    os.replace(tmp, path)
    return path


def great_circle_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(x, dtype=np.float64)) for x in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


__all__ = ["EPOCH", "FetchError", "PluviumError", "atomic_save", "calendar", "fetch", "great_circle_km", "hour_of",
           "hour_to_datetime64", "hour_to_iso", "hub_refusal", "log", "parallel_map", "year_hours"]
