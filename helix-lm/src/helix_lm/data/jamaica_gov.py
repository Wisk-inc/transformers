# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
Jamaica's public record, fetched politely.

Court judgments, Hansard, gazettes, statistics, central-bank reports and government news are the text of how
the country actually works -- the legal and financial Jamaican English that no web crawl has in bulk. This
crawler reads the public sites that publish them, one page at a time per host, obeying ``robots.txt`` and its
crawl delay, identifying itself, and writing what it finds as ``{url, title, text}`` lines under
``data_dir/local/jamaica_gov/`` for ``helix data prepare`` to turn into the ``jamaica_gov`` source.

It is deliberately slow and bounded: a few pages a second across all hosts, a cap per host, a wall-clock cap.
It never follows a link off the host it was given. Newspapers are not here on purpose -- their archives are
copyrighted; public records are not.
"""

from __future__ import annotations

import contextlib
import json
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import urllib.robotparser
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree

USER_AGENT = (
    "CorxLabs-research-crawler/1.0 (+https://huggingface.co/CorxLabs; public records for a Jamaican language model)"
)

# Host -> starting points. Government and court sites; the kind is recorded with every document.
SEEDS: dict[str, dict] = {
    "jis.gov.jm": {"sitemaps": ["https://jis.gov.jm/wp-sitemap.xml"], "kind": "government news"},
    "supremecourt.gov.jm": {"starts": ["https://supremecourt.gov.jm/judgments"], "kind": "judgments"},
    "www.courtofappeal.gov.jm": {"starts": ["https://www.courtofappeal.gov.jm/"], "kind": "judgments"},
    "www.japarliament.gov.jm": {
        "starts": [
            "https://www.japarliament.gov.jm/index.php/publications/hansard",
            "https://www.japarliament.gov.jm/index.php/publications/bills",
            "https://www.japarliament.gov.jm/index.php/publications/acts",
        ],
        "kind": "parliament",
    },
    "moj.gov.jm": {"starts": ["https://moj.gov.jm/"], "kind": "ministry of justice"},
    "laws.moj.gov.jm": {"starts": ["https://laws.moj.gov.jm/"], "kind": "laws"},
    "statinja.gov.jm": {"starts": ["https://statinja.gov.jm/"], "kind": "statistics"},
    "www.pioj.gov.jm": {"starts": ["https://www.pioj.gov.jm/"], "kind": "planning"},
    "boj.org.jm": {"starts": ["https://boj.org.jm/"], "kind": "central bank"},
    "www.mof.gov.jm": {"starts": ["https://www.mof.gov.jm/"], "kind": "finance ministry"},
    "nht.gov.jm": {"starts": ["https://nht.gov.jm/"], "kind": "housing"},
    "www.moh.gov.jm": {"starts": ["https://www.moh.gov.jm/"], "kind": "health ministry"},
    "moey.gov.jm": {"starts": ["https://moey.gov.jm/"], "kind": "education ministry"},
    "opm.gov.jm": {"starts": ["https://opm.gov.jm/"], "kind": "prime minister's office"},
    "www.fscjamaica.org": {"starts": ["https://www.fscjamaica.org/"], "kind": "financial regulator"},
    "www.jamstockex.com": {"starts": ["https://www.jamstockex.com/"], "kind": "stock exchange"},
    "www.odpem.org.jm": {"starts": ["https://www.odpem.org.jm/"], "kind": "disaster management"},
}

SKIP_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".mp4", ".mp3", ".zip", ".rar", ".css", ".js",
                   ".ico", ".woff", ".woff2", ".ttf", ".xls", ".xlsx", ".doc", ".docx", ".ppt", ".pptx")  # fmt: skip
SKIP_PATTERNS = re.compile(
    r"(login|logout|wp-admin|wp-json|\?replytocom|/feed/?$|/tag/|/author/|calendar|/search|share=)", re.I
)
CHROME_TAGS = ("script", "style", "nav", "header", "footer", "noscript", "svg", "form")


class _Text(HTMLParser):
    """Visible text and links from an HTML page; scripts, styles and navigation chrome are dropped."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.links: list[str] = []
        self.title = ""
        self._skip = 0
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in CHROME_TAGS:
            self._skip += 1
        elif tag == "title":
            self._in_title = True
        elif tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.links.append(href)
        elif tag in ("p", "br", "div", "li", "h1", "h2", "h3", "h4", "tr", "td", "section", "article"):
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in CHROME_TAGS and self._skip:
            self._skip -= 1
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self._skip:
            self.parts.append(data)

    def text(self) -> str:
        raw = "".join(self.parts)
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in raw.splitlines()]
        return re.sub(r"\n{3,}", "\n\n", "\n".join(line for line in lines if line)).strip()


def html_to_text(body: bytes) -> tuple[str, str, list[str]]:
    parser = _Text()
    with contextlib.suppress(Exception):
        parser.feed(body.decode("utf-8", "replace"))
    return parser.title.strip(), parser.text(), parser.links


def pdf_to_text(body: bytes) -> str:
    """Text from a PDF through ``pypdf``; an empty string when the library is missing or the file is scanned."""
    try:
        import io

        from pypdf import PdfReader
    except ImportError:
        return ""
    try:
        reader = PdfReader(io.BytesIO(body))
        pages = [page.extract_text() or "" for page in reader.pages[:400]]
    except Exception:
        return ""
    return re.sub(r"\n{3,}", "\n\n", "\n".join(pages)).strip()


@dataclass
class HostCrawl:
    host: str
    seed: dict
    out_dir: Path
    max_pages: int
    deadline: float
    delay: float = 2.0
    log: object = print
    visited: set = field(default_factory=set)
    queue: list = field(default_factory=list)
    pages: int = 0
    documents: int = 0
    robots: urllib.robotparser.RobotFileParser | None = None

    # -- state ---------------------------------------------------------------------------------------
    @property
    def state_path(self) -> Path:
        return self.out_dir / f"{self.host}.state.json"

    @property
    def out_path(self) -> Path:
        return self.out_dir / f"{self.host}.jsonl"

    def load(self) -> None:
        if self.state_path.exists():
            state = json.loads(self.state_path.read_text())
            self.visited, self.queue = set(state["visited"]), state["queue"]
            self.documents = state.get("documents", 0)
        if not self.queue and not self.visited:
            self.queue = list(self.seed.get("starts", []))
            for sitemap in self.seed.get("sitemaps", []):
                self.queue.extend(self._sitemap_urls(sitemap))

    def save(self) -> None:
        self.state_path.write_text(
            json.dumps({"visited": sorted(self.visited), "queue": self.queue[:50_000], "documents": self.documents})
        )

    # -- fetching ------------------------------------------------------------------------------------
    def _fetch(self, url: str, limit: int = 30_000_000) -> tuple[bytes, str] | None:
        request = urllib.request.Request(
            url, headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/pdf"}
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                kind = response.headers.get("Content-Type", "").split(";")[0].strip().lower()
                return response.read(limit), kind
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, ValueError):
            return None

    def _allowed(self, url: str) -> bool:
        if self.robots is None:
            self.robots = urllib.robotparser.RobotFileParser()
            scheme = "http" if url.startswith("http://") else "https"
            self.robots.set_url(f"{scheme}://{self.host}/robots.txt")
            fetched = self._fetch(self.robots.url)
            self.robots.parse((fetched[0].decode("utf-8", "replace") if fetched else "").splitlines())
            with contextlib.suppress(Exception):
                crawl_delay = self.robots.crawl_delay(USER_AGENT) or self.robots.crawl_delay("*")
                if crawl_delay:
                    self.delay = max(self.delay, float(crawl_delay))
        try:
            return self.robots.can_fetch(USER_AGENT, url) and self.robots.can_fetch("*", url)
        except Exception:
            return True

    def _sitemap_urls(self, sitemap: str, depth: int = 0) -> list[str]:
        fetched = self._fetch(sitemap)
        if not fetched or depth > 3:
            return []
        try:
            root = ElementTree.fromstring(fetched[0])
        except ElementTree.ParseError:
            return []
        urls = []
        for element in root.iter():
            if element.tag.endswith("loc") and element.text:
                location = element.text.strip()
                if location.endswith(".xml") and "sitemap" in location:
                    urls.extend(self._sitemap_urls(location, depth + 1))
                else:
                    urls.append(location)
        return urls

    def _keep_link(self, link: str, base: str) -> str | None:
        absolute = urllib.parse.urljoin(base, link.split("#")[0])
        parsed = urllib.parse.urlparse(absolute)
        if parsed.scheme not in ("http", "https") or parsed.netloc != self.host:
            return None
        if parsed.path.lower().endswith(SKIP_EXTENSIONS) or SKIP_PATTERNS.search(absolute):
            return None
        return absolute

    # -- the loop ------------------------------------------------------------------------------------
    def run(self) -> int:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.load()
        with open(self.out_path, "a", encoding="utf-8") as out:
            while self.queue and self.pages < self.max_pages and time.time() < self.deadline:
                url = self.queue.pop(0)
                if url in self.visited:
                    continue
                self.visited.add(url)
                if not self._allowed(url):
                    continue
                fetched = self._fetch(url)
                self.pages += 1
                time.sleep(self.delay)
                if not fetched:
                    continue
                body, kind = fetched
                if "pdf" in kind or url.lower().endswith(".pdf"):
                    title, text, links = Path(urllib.parse.urlparse(url).path).name, pdf_to_text(body), []
                elif "html" in kind or kind == "":
                    title, text, links = html_to_text(body)
                else:
                    continue
                for link in links:
                    kept = self._keep_link(link, url)
                    if kept and kept not in self.visited:
                        self.queue.append(kept)
                if len(text) >= 300:
                    record = {"url": url, "title": title, "text": text, "kind": self.seed.get("kind", ""),
                              "fetched": time.strftime("%Y-%m-%d")}  # fmt: skip
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    self.documents += 1
                if self.pages % 50 == 0:
                    out.flush()
                    self.save()
                    self.log(
                        f"[crawl] {self.host}: {self.pages} pages this run, {self.documents} documents, "
                        f"{len(self.queue)} queued"
                    )
        self.save()
        return self.documents


def crawl(
    data_dir: str | Path,
    hosts: dict[str, dict] | None = None,
    hours: float = 6.0,
    max_pages_per_host: int = 20_000,
    parallel_hosts: int = 6,
    delay: float = 2.0,
    log=print,
) -> dict[str, int]:
    """Crawl every seed host in parallel (one request at a time per host) for at most ``hours``."""
    out_dir = Path(data_dir) / "local" / "jamaica_gov"
    deadline = time.time() + hours * 3600
    hosts = hosts or SEEDS
    results: dict[str, int] = {}
    lock = threading.Lock()

    def one(host: str) -> None:
        crawler = HostCrawl(host, hosts[host], out_dir, max_pages_per_host, deadline, delay=delay, log=log)
        try:
            count = crawler.run()
        except Exception as error:  # one broken site must not end the crawl
            log(f"[crawl] {host}: {type(error).__name__}: {str(error)[:120]}")
            count = crawler.documents
        with lock:
            results[host] = count

    with ThreadPoolExecutor(max_workers=parallel_hosts) as pool:
        list(pool.map(one, list(hosts)))
    total = sum(results.values())
    log(f"[crawl] done: {total} documents from {len(results)} hosts in {out_dir}")
    (out_dir / "summary.json").write_text(
        json.dumps({"documents": results, "finished": time.time() >= deadline}, indent=1)
    )
    return results
