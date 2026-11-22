# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
Pictures to learn from, and where they may be taken from.

**Paintings** come from three museums that publish their public-domain works for anyone to use, with an API meant
for programs:

* the Art Institute of Chicago (public-domain works, CC0; images over IIIF),
* the Cleveland Museum of Art (Open Access, CC0),
* the National Gallery of Art, Washington (Open Access, CC0; images over IIIF).

The Metropolitan Museum's open-access pictures are not used: its image server answers programs with a bot check.
Every painting is recorded in ``paintings.jsonl`` with its title, artist, museum and licence.

**Photos** come from Open Images V7's validation set (photos their uploaders licensed CC BY 2.0; the author and
licence of each are kept in ``photos.jsonl``, which a CC BY work needs), or from any folder of your own.

A download can be stopped at any file and goes on where it stopped: a picture already on disk is not fetched
again, and a file is only given its name once it is whole and opens as a picture.
"""

from __future__ import annotations

import csv
import io
import json
import os
import random
import urllib.parse
import urllib.request
from pathlib import Path

import torch

AGENT = {"User-Agent": "helix-lm-pinxit/0.1 (open-access art for model training; github.com/Wisk-inc/transformers)",
         "AIC-User-Agent": "helix-lm-pinxit/0.1 (github.com/Wisk-inc/transformers)"}
SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")
OPEN_IMAGES = "https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv"


def fetch(url: str, timeout: float = 60.0) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=AGENT), timeout=timeout) as response:
        return response.read()


def chicago(limit: int, fetch_=fetch):
    """Public-domain paintings of the Art Institute of Chicago."""
    page = 1
    while limit > 0:
        query = {"query[bool][must][0][term][is_public_domain]": "true",
                 "query[bool][must][1][term][artwork_type_title.keyword]": "Painting",
                 "fields": "id,title,artist_title,date_display,image_id", "limit": 100, "page": page}
        data = json.loads(fetch_("https://api.artic.edu/api/v1/artworks/search?" + urllib.parse.urlencode(query)))
        rows = data.get("data", [])
        if not rows:
            return
        for row in rows:
            if row.get("image_id") and limit > 0:
                limit -= 1
                yield {"id": f"aic-{row['id']}", "title": row.get("title") or "", "artist": row.get("artist_title")
                       or "", "date": row.get("date_display") or "", "museum": "Art Institute of Chicago",
                       "licence": "CC0 (public domain)",
                       "url": f"https://www.artic.edu/iiif/2/{row['image_id']}/full/843,/0/default.jpg"}
        page += 1
        if page > 10:                                   # the search API serves 1,000 results at most
            return


def cleveland(limit: int, fetch_=fetch):
    """CC0 paintings of the Cleveland Museum of Art."""
    skip = 0
    while limit > 0:
        query = {"cc0": 1, "has_image": 1, "type": "Painting", "limit": 100, "skip": skip,
                 "fields": "id,title,creators,creation_date,images,share_license_status"}
        data = json.loads(fetch_("https://openaccess-api.clevelandart.org/api/artworks/?"
                                 + urllib.parse.urlencode(query)))
        rows = data.get("data", [])
        if not rows:
            return
        for row in rows:
            url = ((row.get("images") or {}).get("web") or {}).get("url")
            if url and row.get("share_license_status") == "CC0" and limit > 0:
                limit -= 1
                artist = (row.get("creators") or [{}])[0].get("description", "")
                yield {"id": f"cma-{row['id']}", "title": row.get("title") or "", "artist": artist,
                       "date": row.get("creation_date") or "", "museum": "Cleveland Museum of Art",
                       "licence": "CC0", "url": url}
        skip += 100


def washington(limit: int, fetch_=fetch):
    """Open-access paintings of the National Gallery of Art."""
    base = "https://raw.githubusercontent.com/NationalGalleryOfArt/opendata/main/data/"
    objects = {}
    for row in csv.DictReader(io.StringIO(fetch_(base + "objects.csv", 300).decode("utf-8", "replace"))):
        if row.get("classification") == "Painting":
            objects[row["objectid"]] = row
    for row in csv.DictReader(io.StringIO(fetch_(base + "published_images.csv", 300).decode("utf-8", "replace"))):
        if limit <= 0:
            return
        obj = objects.get(row.get("depictstmsobjectid"))
        if obj and row.get("openaccess") == "1" and row.get("viewtype") == "primary":
            limit -= 1
            yield {"id": f"nga-{obj['objectid']}", "title": obj.get("title") or "", "artist": obj.get("attribution")
                   or "", "date": obj.get("displaydate") or "", "museum": "National Gallery of Art, Washington",
                   "licence": "CC0 (open access)", "url": row["iiifurl"] + "/full/!1024,1024/0/default.jpg"}


def open_images(limit: int, fetch_=fetch):
    """Photos of Open Images V7's validation set, with their authors and licences."""
    text = fetch_(OPEN_IMAGES, 300).decode("utf-8", "replace")
    for row in csv.DictReader(io.StringIO(text)):
        if limit <= 0:
            return
        if "creativecommons.org/licenses/by/2.0" in (row.get("License") or ""):
            limit -= 1
            yield {"id": f"oi-{row['ImageID']}", "title": row.get("Title") or "", "artist": row.get("Author") or "",
                   "licence": "CC BY 2.0", "page": row.get("OriginalLandingURL") or "",
                   "url": row.get("Thumbnail300KURL") or row.get("OriginalURL") or "",
                   "second_url": f"https://s3.amazonaws.com/open-images-dataset/validation/{row['ImageID']}.jpg"}


SOURCES = {"chicago": chicago, "cleveland": cleveland, "washington": washington}


def download(items, folder, record: str, max_side: int = 768, should_stop=None, log=print, fetch_=fetch,
             workers: int = 6) -> int:
    """Fetch pictures into ``folder`` (each shrunk to ``max_side`` and saved as JPEG, ``workers`` at a time) and
    add a line about each to ``folder/record``. A picture already there is skipped; one that cannot be fetched
    or opened is said and left out. ``should_stop()`` True stops before the next picture. Returns how many
    pictures the folder holds."""
    from concurrent.futures import ThreadPoolExecutor

    from PIL import Image

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    listed = set()
    record_path = folder / record
    if record_path.exists():
        for line in record_path.read_text().splitlines():
            try:
                listed.add(json.loads(line)["id"])
            except (ValueError, KeyError):
                continue                                  # a line cut short by a stop

    def one(item: dict):
        path = folder / f"{item['id']}.jpg"
        if path.exists():
            return item, None
        problem = "no address"
        for url in (item.get("url"), item.get("second_url")):
            if not url:
                continue
            try:
                picture = Image.open(io.BytesIO(fetch_(url)))
                picture.load()
                picture = picture.convert("RGB")
                picture.thumbnail((max_side, max_side))
                tmp = path.with_name(f"{path.name}.{os.getpid()}.{id(item)}.tmp")
                picture.save(tmp, "JPEG", quality=92)
                os.replace(tmp, path)
                return item, None
            except Exception as error:                    # the next address, or the next picture
                problem = f"{type(error).__name__}: {str(error)[:80]}"
        return item, problem

    def wanted():
        for item in items:
            if should_stop is not None and should_stop():
                return
            yield item

    new = failed = 0
    with record_path.open("a") as out, ThreadPoolExecutor(max(1, workers)) as pool:
        for item, problem in pool.map(one, wanted()):
            if problem is not None:
                failed += 1
                if failed <= 5:
                    log(f"  {item['id']}: not fetched ({problem})")
                continue
            if item["id"] not in listed:
                out.write(json.dumps(item) + "\n")
                out.flush()
                listed.add(item["id"])
                new += 1
                if new % 500 == 0:
                    log(f"  {folder.name}: {new} more pictures listed")
    if failed > 5:
        log(f"  {folder.name}: {failed} pictures could not be fetched")
    return len(list(folder.glob("*.jpg")))


def find_pictures(folder) -> list[Path]:
    folder = Path(folder)
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in SUFFIXES) if folder.exists() else []


def style_cards(folder, log=print, should_stop=None) -> int:
    """Measure every painting in ``folder`` once and keep its card beside it (``<name>.card.json``)."""
    from .style import read_style

    made = 0
    for path in find_pictures(folder):
        card = path.with_suffix(".card.json")
        if card.exists():
            continue
        if should_stop is not None and should_stop():
            break
        try:
            tmp = card.with_name(f"{card.name}.{os.getpid()}.tmp")
            tmp.write_text(json.dumps(read_style(path, name=path.stem).to_json()))
            os.replace(tmp, card)
            made += 1
        except Exception as error:                        # a picture that cannot be read is not a style
            log(f"  {path.name}: no card ({type(error).__name__}: {str(error)[:80]})")
    return made


class PlanStream(torch.utils.data.IterableDataset):
    """Plans without end for :func:`helix_lm.paint.train.train_pictures`: a photo, a painting's card (or the
    photo's own colours, one time in five), a feeling (the card's own, one of the eight, or none), and the mind's
    settings varied a little, so the model meets every kind of brief."""

    def __init__(self, photos: list, paintings: list, work: int = 256, seed: int = 0) -> None:
        self.photos, self.paintings, self.work, self.seed = list(photos), list(paintings), work, seed
        if not self.photos:
            raise ValueError("no photos to train on")

    def __iter__(self):
        from .emotion import EMOTIONS, feeling_of
        from .io import load_image
        from .mind import Settings, make_plan
        from .style import StyleCard, own_card

        info = torch.utils.data.get_worker_info()
        rng = random.Random(self.seed * 1000 + (info.id if info else 0) + int.from_bytes(os.urandom(2), "little"))
        while True:
            try:
                photo = load_image(rng.choice(self.photos), max_side=2 * self.work)
                if min(photo.shape[-2:]) < 64:
                    continue
                card = None
                if self.paintings and rng.random() < 0.8:
                    side = Path(rng.choice(self.paintings)).with_suffix(".card.json")
                    card = StyleCard.from_json(json.loads(side.read_text())) if side.exists() else None
                card = card or own_card(photo, seed=rng.randrange(1 << 20))
                roll = rng.random()
                if roll < 0.4 and card.emotion:
                    feeling = feeling_of(card.emotion, 0.7)
                else:
                    feeling = feeling_of(rng.choice(EMOTIONS), rng.uniform(0.5, 1.0)) if roll < 0.85 else \
                        feeling_of(None)
                settings = Settings(work=self.work, compose=rng.choice((0.0, 0.6, 1.0)),
                                    exaggerate=rng.uniform(0.0, 1.0), restraint=rng.choice((None, rng.random())),
                                    style_strength=rng.uniform(0.4, 1.0), seed=rng.randrange(1 << 20))
                yield make_plan(photo, card, feeling, settings)
            except (OSError, ValueError, RuntimeError):   # a picture that cannot be read: the next one
                continue


def plan_stream(photos: list, paintings: list, work: int = 256, workers: int = 4, seed: int = 0):
    """An iterator of plans made by ``workers`` processes."""
    stream = PlanStream(photos, paintings, work, seed)
    if workers <= 0:
        return iter(stream)
    return iter(torch.utils.data.DataLoader(stream, batch_size=None, num_workers=workers, prefetch_factor=4,
                                            persistent_workers=True))


__all__ = ["AGENT", "SOURCES", "PlanStream", "chicago", "cleveland", "download", "fetch", "find_pictures",
           "open_images", "plan_stream", "style_cards", "washington"]
