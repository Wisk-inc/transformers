# Copyright 2026 Nathan. Apache-2.0.
"""
A minimal reader for consolidated Zarr v2 stores over HTTPS (the public ERA5 stores on Google Cloud).

``xarray + zarr + gcsfs`` can open these stores, but each one opens a connection pool and an event loop, pulls
in a stack of dependencies, and reads whole chunks only to throw away 98% of them when the Caribbean is wanted.
This reader does the one thing needed: fetch the chunks that cover a time range, decompress each with numcodecs,
crop it to a box at once, and stack the crops. Missing chunks read as the array's fill value.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import numpy as np

from .util import fetch, parallel_map


def _codec(config: dict | None):
    if not config:
        return None
    import numcodecs

    return numcodecs.get_codec(dict(config))


@dataclass
class ZarrArray:
    base: str
    name: str
    shape: tuple
    chunks: tuple
    dtype: np.dtype
    compressor: object
    filters: list
    fill_value: object
    order: str
    separator: str
    attrs: dict = field(default_factory=dict)

    @property
    def dims(self) -> list:
        return list(self.attrs.get("_ARRAY_DIMENSIONS", []))

    def chunk_url(self, index: tuple) -> str:
        return f"{self.base}/{self.name}/" + self.separator.join(str(int(i)) for i in index)

    def decode(self, raw: bytes | None) -> np.ndarray:
        if raw is None:
            fill = self.fill_value if self.fill_value is not None else 0
            if isinstance(fill, str):  # "NaN" in JSON metadata
                fill = float(fill)
            return np.full(self.chunks, fill, dtype=self.dtype)
        data = self.compressor.decode(raw) if self.compressor is not None else raw
        for codec in reversed(self.filters):
            data = codec.decode(data)
        return np.frombuffer(data, dtype=self.dtype).reshape(self.chunks, order=self.order)

    def read_chunk(self, index: tuple) -> np.ndarray:
        return self.decode(fetch(self.chunk_url(index), missing_ok=True))

    def read(self, selection: tuple, workers: int = 16, progress: str | None = None) -> np.ndarray:
        """
        ``selection`` holds one ``slice`` (step 1) or ``int`` per dimension. Only the chunks it touches are fetched,
        each cropped as soon as it arrives, and the crops are assembled in order.
        """
        sel = []
        for size, s in zip(self.shape, selection):
            if isinstance(s, (int, np.integer)):
                s = slice(int(s), int(s) + 1)
            start, stop, step = s.indices(size)
            if step != 1:
                raise ValueError("only unit-step slices are supported")
            sel.append((start, stop))
        ranges = [range(a // c, max(a // c, (b - 1) // c) + 1) if b > a else range(0) for (a, b), c in zip(sel, self.chunks)]
        indices = [()]
        for r in ranges:
            indices = [i + (j,) for i in indices for j in r]

        def one(index):
            block = self.read_chunk(index)
            crop = []
            for (a, b), c, i in zip(sel, self.chunks, index):
                lo, hi = max(a, i * c), min(b, (i + 1) * c)
                crop.append(slice(lo - i * c, hi - i * c))
            return index, np.ascontiguousarray(block[tuple(crop)])

        out = np.empty([b - a for a, b in sel], dtype=self.dtype)
        for index, crop in parallel_map(one, indices, workers=workers, progress=progress):
            where = []
            for (a, b), c, i in zip(sel, self.chunks, index):
                lo = max(a, i * c)
                where.append(slice(lo - a, lo - a + crop.shape[len(where)]))
            out[tuple(where)] = crop
        return out


class ZarrStore:
    """``ZarrStore(url)["total_precipitation"].read((slice(t0, t1), slice(None), slice(None)))``."""

    def __init__(self, url: str) -> None:
        self.url = url.rstrip("/")
        self.metadata = json.loads(fetch(self.url + "/.zmetadata"))["metadata"]
        self.attrs = self.metadata.get(".zattrs", {})

    @property
    def names(self) -> list:
        return sorted(k[: -len("/.zarray")] for k in self.metadata if k.endswith("/.zarray"))

    def __contains__(self, name: str) -> bool:
        return f"{name}/.zarray" in self.metadata

    def __getitem__(self, name: str) -> ZarrArray:
        meta = self.metadata[f"{name}/.zarray"]
        return ZarrArray(
            base=self.url, name=name, shape=tuple(meta["shape"]), chunks=tuple(meta["chunks"]),
            dtype=np.dtype(meta["dtype"]), compressor=_codec(meta.get("compressor")),
            filters=[_codec(f) for f in (meta.get("filters") or [])], fill_value=meta.get("fill_value"),
            order=meta.get("order", "C"), separator=meta.get("dimension_separator", "."),
            attrs=self.metadata.get(f"{name}/.zattrs", {}),
        )  # fmt: skip

    def coordinate(self, name: str) -> np.ndarray:
        array = self[name]
        return array.read(tuple(slice(None) for _ in array.shape))

    def hours(self, name: str = "time") -> np.ndarray:
        """The time coordinate as integer hours since 1970 (decoding the CF ``units`` attribute)."""
        array = self[name]
        return decode_cf_hours(self.coordinate(name), array.attrs.get("units", "hours since 1970-01-01"))


def decode_cf_hours(values: np.ndarray, units: str) -> np.ndarray:
    """CF ``"<unit> since <date>"`` times as integer hours since 1970-01-01 (rounded to the hour)."""
    unit, _, origin = units.partition(" since ")
    origin = origin.strip().replace(" ", "T")
    if len(origin) == 10:
        origin += "T00"
    base = np.datetime64(origin[:19] if len(origin) > 13 else origin, "s")
    scale = {"seconds": 1, "second": 1, "minutes": 60, "minute": 60, "hours": 3600, "hour": 3600, "days": 86400,
             "day": 86400}[unit.strip().lower()]
    seconds = np.asarray(values, dtype=np.float64) * scale
    epoch = (base - np.datetime64("1970-01-01T00:00:00", "s")).astype(np.int64)
    return np.round((epoch + seconds) / 3600.0).astype(np.int64)


def axis_window(axis: np.ndarray, low: float, high: float, wrap: float | None = None) -> slice:
    """The contiguous index range of a monotonic coordinate ``axis`` covering ``[low, high]`` (``wrap`` = 360 maps
    western longitudes onto a 0-360 axis)."""
    axis = np.asarray(axis, dtype=np.float64)
    if wrap is not None and axis.min() >= 0 and low < 0:
        low, high = low % wrap, high % wrap
    inside = np.nonzero((axis >= min(low, high) - 1e-9) & (axis <= max(low, high) + 1e-9))[0]
    if inside.size == 0:
        raise ValueError(f"axis has no values in [{low}, {high}]")
    return slice(int(inside.min()), int(inside.max()) + 1)


def chunk_bytes(array: ZarrArray) -> int:
    return int(math.prod(array.chunks) * array.dtype.itemsize)


__all__ = ["ZarrArray", "ZarrStore", "axis_window", "chunk_bytes", "decode_cf_hours"]
