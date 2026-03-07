from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Document:
    content: str
    metadata: dict = field(default_factory=dict)
    format: str = "unknown"


def load_file(path: str | Path) -> Document:
    """Dispatch to the appropriate loader based on file extension."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".html", ".htm"):
        return _load_html(path)
    elif suffix == ".md":
        return _load_markdown(path)
    elif suffix == ".pdf":
        return _load_pdf(path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def _load_html(path: Path) -> Document:
    from bs4 import BeautifulSoup
    text = path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(text, "html.parser")
    title = soup.title.string.strip() if soup.title and soup.title.string else path.stem
    content = soup.get_text(separator="\n", strip=True)
    return Document(
        content=content,
        metadata={"source": str(path), "title": title, "format": "html"},
        format="html",
    )


def _load_markdown(path: Path) -> Document:
    content = path.read_text(encoding="utf-8", errors="replace")
    title = path.stem
    for line in content.splitlines():
        if line.startswith("# "):
            title = line.lstrip("# ").strip()
            break
    return Document(
        content=content,
        metadata={"source": str(path), "title": title, "format": "markdown"},
        format="markdown",
    )


def _load_pdf(path: Path) -> Document:
    from unstructured.partition.pdf import partition_pdf
    elements = partition_pdf(filename=str(path))
    content = "\n\n".join(str(e) for e in elements)
    return Document(
        content=content,
        metadata={"source": str(path), "title": path.stem, "format": "pdf"},
        format="pdf",
    )
