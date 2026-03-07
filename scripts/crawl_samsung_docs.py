#!/usr/bin/env python
"""Crawler for Samsung Smart TV developer documentation.

Downloads HTML pages from developer.samsung.com/smarttv into
data/raw/samsung-tv/ for later ingestion via scripts/ingest.py.

Usage:
    python scripts/crawl_samsung_docs.py
    python scripts/crawl_samsung_docs.py --limit 300 --delay 1.0
    python scripts/crawl_samsung_docs.py --output /custom/path --depth 3
"""

from __future__ import annotations

import argparse
import time
import urllib.parse
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Seed URLs — one entry point per major documentation section
# ---------------------------------------------------------------------------
SEED_URLS: list[str] = [
    # Getting started
    "https://developer.samsung.com/smarttv/develop/getting-started/quick-start-guide.html",
    # Web app guides
    "https://developer.samsung.com/smarttv/develop/guides.html",
    # Samsung Product API (TV-specific extensions)
    "https://developer.samsung.com/smarttv/develop/api-references/samsung-product-api-references.html",
    # Tizen Web Device APIs
    "https://developer.samsung.com/smarttv/develop/api-references/tizen-web-device-api-references.html",
    # config.xml / manifest
    "https://developer.samsung.com/smarttv/develop/getting-started/creating-your-first-samsung-smart-tv-web-application.html",
    # Tools & SDK
    "https://developer.samsung.com/smarttv/develop/tools.html",
    # User interaction / remote control
    "https://developer.samsung.com/smarttv/develop/guides/user-interaction.html",
]

_ALLOWED_PREFIX = "https://developer.samsung.com/smarttv"
_USER_AGENT = "Mozilla/5.0 (compatible; DevDocsRagBot/1.0; +https://github.com/devdocs-rag)"
_ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fetch(url: str, timeout: int = 15) -> bytes | None:
    """Download a URL and return its bytes, or None if non-HTML / error."""
    req = Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urlopen(req, timeout=timeout) as resp:
            content_type = resp.headers.get("Content-Type", "")
            if "text/html" not in content_type:
                return None
            return resp.read()
    except (URLError, HTTPError, OSError) as exc:
        print(f"  ! {exc}")
        return None


def _extract_links(html: bytes, base_url: str) -> list[str]:
    """Return all in-scope HTML links found in the page."""
    soup = BeautifulSoup(html, "html.parser")
    links: list[str] = []
    for tag in soup.find_all("a", href=True):
        href = tag["href"].strip()
        full = urllib.parse.urljoin(base_url, href)
        parsed = urllib.parse.urlparse(full)
        # Strip fragment and query string — we want canonical page URLs
        clean = urllib.parse.urlunparse(parsed._replace(fragment="", query=""))
        if clean.startswith(_ALLOWED_PREFIX) and clean.endswith((".html", ".htm")):
            links.append(clean)
    return links


def _url_to_filename(url: str) -> str:
    """Map a URL to a flat filename safe for any filesystem."""
    parsed = urllib.parse.urlparse(url)
    # e.g. /smarttv/develop/guides/user-interaction.html
    # → smarttv__develop__guides__user-interaction.html
    path = parsed.path.strip("/").replace("/", "__")
    if not path.endswith((".html", ".htm")):
        path += ".html"
    return path


# ---------------------------------------------------------------------------
# Crawler
# ---------------------------------------------------------------------------

def crawl(
    seed_urls: list[str],
    output_dir: Path,
    max_pages: int = 500,
    max_depth: int = 4,
    delay: float = 0.5,
) -> int:
    """BFS crawl starting from seed_urls. Returns the number of pages saved."""
    output_dir.mkdir(parents=True, exist_ok=True)

    visited: set[str] = set()
    # Queue entries: (url, depth)
    queue: list[tuple[str, int]] = [(u, 0) for u in seed_urls]
    saved = 0

    while queue and saved < max_pages:
        url, depth = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)

        print(f"[{saved + 1:>4}] depth={depth}  {url}")
        html = _fetch(url)
        if html is None:
            print("       skipped")
            continue

        filename = _url_to_filename(url)
        (output_dir / filename).write_bytes(html)
        saved += 1

        if depth < max_depth:
            for link in _extract_links(html, url):
                if link not in visited:
                    queue.append((link, depth + 1))

        time.sleep(delay)

    return saved


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crawl Samsung Smart TV developer docs to data/raw/samsung-tv/",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output",
        default=str(_ROOT / "data" / "raw" / "samsung-tv"),
        help="Directory to write HTML files",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Maximum number of pages to download",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=4,
        help="Maximum BFS depth from seed URLs",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Seconds to wait between requests",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    print(f"Output : {output_dir}")
    print(f"Limit  : {args.limit} pages")
    print(f"Depth  : {args.depth}")
    print(f"Delay  : {args.delay}s\n")

    n = crawl(
        seed_urls=SEED_URLS,
        output_dir=output_dir,
        max_pages=args.limit,
        max_depth=args.depth,
        delay=args.delay,
    )

    print(f"\nDone — {n} pages saved to {output_dir}")
    print(f"\nNext step — ingest into ChromaDB:")
    print(f"  python scripts/ingest.py {output_dir} --collection samsung_tv --doc-type guide")


if __name__ == "__main__":
    main()
