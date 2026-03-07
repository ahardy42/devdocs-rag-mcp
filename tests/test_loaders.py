"""Tests for document loaders (Step 5)."""

import pytest

from devdocs_rag.ingest.loaders import Document, load_file

_HTML_WITH_TITLE = """\
<html>
<head><title>Samsung TV Guide</title></head>
<body>
<h1>Remote Control</h1>
<p>Handle remote input events on Samsung TV.</p>
</body>
</html>"""

_HTML_WITHOUT_TITLE = """\
<html>
<body>
<h1>Introduction</h1>
<p>Some content here.</p>
</body>
</html>"""

_MARKDOWN_WITH_HEADING = """\
# Getting Started

Install the Tizen SDK to start developing Samsung TV apps.

## Configuration

Edit config.xml to configure your web application."""

_MARKDOWN_WITHOUT_HEADING = """\
Some text without a heading.
More content here."""


# ---------------------------------------------------------------------------
# HTML loader
# ---------------------------------------------------------------------------

def test_load_html_returns_document(tmp_path):
    f = tmp_path / "guide.html"
    f.write_text(_HTML_WITH_TITLE, encoding="utf-8")
    assert isinstance(load_file(f), Document)


def test_load_html_format(tmp_path):
    f = tmp_path / "guide.html"
    f.write_text(_HTML_WITH_TITLE, encoding="utf-8")
    assert load_file(f).format == "html"


def test_load_html_extracts_title(tmp_path):
    f = tmp_path / "guide.html"
    f.write_text(_HTML_WITH_TITLE, encoding="utf-8")
    assert load_file(f).metadata["title"] == "Samsung TV Guide"


def test_load_html_title_falls_back_to_stem(tmp_path):
    f = tmp_path / "my_guide.html"
    f.write_text(_HTML_WITHOUT_TITLE, encoding="utf-8")
    assert load_file(f).metadata["title"] == "my_guide"


def test_load_html_content_has_no_tags(tmp_path):
    f = tmp_path / "guide.html"
    f.write_text(_HTML_WITH_TITLE, encoding="utf-8")
    content = load_file(f).content
    assert "<html>" not in content
    assert "<h1>" not in content


def test_load_html_content_has_text(tmp_path):
    f = tmp_path / "guide.html"
    f.write_text(_HTML_WITH_TITLE, encoding="utf-8")
    assert "Remote Control" in load_file(f).content


def test_load_html_metadata_has_source(tmp_path):
    f = tmp_path / "guide.html"
    f.write_text(_HTML_WITH_TITLE, encoding="utf-8")
    assert "source" in load_file(f).metadata


def test_load_htm_extension(tmp_path):
    f = tmp_path / "page.htm"
    f.write_text(_HTML_WITH_TITLE, encoding="utf-8")
    assert load_file(f).format == "html"


# ---------------------------------------------------------------------------
# Markdown loader
# ---------------------------------------------------------------------------

def test_load_markdown_returns_document(tmp_path):
    f = tmp_path / "guide.md"
    f.write_text(_MARKDOWN_WITH_HEADING, encoding="utf-8")
    assert isinstance(load_file(f), Document)


def test_load_markdown_format(tmp_path):
    f = tmp_path / "guide.md"
    f.write_text(_MARKDOWN_WITH_HEADING, encoding="utf-8")
    assert load_file(f).format == "markdown"


def test_load_markdown_extracts_title_from_heading(tmp_path):
    f = tmp_path / "guide.md"
    f.write_text(_MARKDOWN_WITH_HEADING, encoding="utf-8")
    assert load_file(f).metadata["title"] == "Getting Started"


def test_load_markdown_title_falls_back_to_stem(tmp_path):
    f = tmp_path / "my_guide.md"
    f.write_text(_MARKDOWN_WITHOUT_HEADING, encoding="utf-8")
    assert load_file(f).metadata["title"] == "my_guide"


def test_load_markdown_content_preserved(tmp_path):
    f = tmp_path / "guide.md"
    f.write_text(_MARKDOWN_WITH_HEADING, encoding="utf-8")
    content = load_file(f).content
    assert "Getting Started" in content
    assert "Tizen SDK" in content


def test_load_markdown_metadata_has_source(tmp_path):
    f = tmp_path / "guide.md"
    f.write_text(_MARKDOWN_WITH_HEADING, encoding="utf-8")
    assert "source" in load_file(f).metadata


# ---------------------------------------------------------------------------
# Unsupported formats
# ---------------------------------------------------------------------------

def test_unsupported_extension_raises(tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text("some text", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported"):
        load_file(f)
