#!/usr/bin/env python
"""General-purpose recursive website crawler.

Downloads HTML pages starting from one or more seed URLs into
data/raw/<directory>/ for later ingestion via scripts/ingest.py.

Usage:
    crawl-docs my-site https://example.com/docs
    crawl-docs my-site https://example.com/a,https://example.com/b
    crawl-docs my-site https://example.com/docs --allowed-prefix https://example.com/docs
    crawl-docs my-site https://example.com/docs --limit 200 --delay 1.0 --depth 3
"""

from __future__ import annotations

import argparse
import time
import urllib.parse
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup

_USER_AGENT = "Mozilla/5.0 (compatible; DevDocsRagBot/1.0; +https://github.com/devdocs-rag)"
_ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _domain_prefix(url: str) -> str:
    """Return scheme://netloc for a URL."""
    parsed = urllib.parse.urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}"


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


def _extract_links(html: bytes, base_url: str, allowed_prefixes: list[str]) -> list[str]:
    """Return all in-scope HTML links found in the page."""
    soup = BeautifulSoup(html, "html.parser")
    links: list[str] = []
    for tag in soup.find_all("a", href=True):
        href = tag["href"].strip()
        full = urllib.parse.urljoin(base_url, href)
        parsed = urllib.parse.urlparse(full)
        # Strip fragment and query string — canonical page URLs only
        clean = urllib.parse.urlunparse(parsed._replace(fragment="", query=""))
        if any(clean.startswith(prefix) for prefix in allowed_prefixes):
            links.append(clean)
    return links


def _url_to_filename(url: str) -> str:
    """Map a URL to a flat filename safe for any filesystem."""
    parsed = urllib.parse.urlparse(url)
    # Strip leading slash and replace path separators with double underscores
    path = parsed.path.strip("/").replace("/", "__")
    if not path:
        path = "index"
    if not path.endswith((".html", ".htm")):
        path += ".html"
    return path


# ---------------------------------------------------------------------------
# Crawler
# ---------------------------------------------------------------------------

def crawl(
    seed_urls: list[str],
    output_dir: Path,
    allowed_prefixes: list[str],
    max_pages: int = 500,
    max_depth: int = 4,
    delay: float = 0.5,
) -> int:
    """BFS crawl starting from seed_urls. Returns the number of pages saved."""
    output_dir.mkdir(parents=True, exist_ok=True)

    visited: set[str] = set()
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
            for link in _extract_links(html, url, allowed_prefixes):
                if link not in visited:
                    queue.append((link, depth + 1))

        time.sleep(delay)

    return saved


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recursively crawl websites and save HTML to data/raw/<directory>/",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "directory",
        help="Output collection name (saved to data/raw/<directory>/)",
    )
    parser.add_argument(
        "seed_urls",
        help="Comma-separated seed URLs to start crawling from",
    )
    parser.add_argument(
        "--allowed-prefix",
        default=None,
        help=(
            "Only follow links whose URL starts with this prefix. "
            "Defaults to the domain(s) of the seed URLs."
        ),
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

    seeds = [u.strip() for u in args.seed_urls.split(",") if u.strip()]
    if not seeds:
        parser.error("At least one seed URL is required")

    if args.allowed_prefix:
        allowed_prefixes = [args.allowed_prefix]
    else:
        # Default: allow all domains present in the seed URLs
        seen: dict[str, None] = {}
        for u in seeds:
            seen[_domain_prefix(u)] = None
        allowed_prefixes = list(seen)

    output_dir = _ROOT / "data" / "raw" / args.directory

    print(f"Directory      : {args.directory}")
    print(f"Output         : {output_dir}")
    print(f"Seeds          : {seeds}")
    print(f"Allowed prefix : {allowed_prefixes}")
    print(f"Limit          : {args.limit} pages")
    print(f"Depth          : {args.depth}")
    print(f"Delay          : {args.delay}s\n")

    n = crawl(
        seed_urls=seeds,
        output_dir=output_dir,
        allowed_prefixes=allowed_prefixes,
        max_pages=args.limit,
        max_depth=args.depth,
        delay=args.delay,
    )

    print(f"\nDone — {n} pages saved to {output_dir}")
    print(f"\nNext step — ingest into ChromaDB:")
    print(f"  python scripts/ingest.py {output_dir} --collection {args.directory}")


if __name__ == "__main__":
    main()
