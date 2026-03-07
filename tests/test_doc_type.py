"""Tests for doc_type inference (ingest/doc_type.py)."""

from pathlib import Path

import pytest

from devdocs_rag.ingest.doc_type import infer_doc_type


# ---------------------------------------------------------------------------
# Samsung TV __ encoded filenames
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename,expected", [
    ("smarttv__develop__api-references__avplay-api.html",             "api_reference"),
    ("smarttv__develop__api-references__overview.html",               "api_reference"),
    ("smarttv__develop__api-references__tizen-web-device-api-references__application-api.html", "api_reference"),
    ("smarttv__design__input-methods.html",                           "design"),
    ("smarttv__design__ux-checklist.html",                            "design"),
    ("smarttv__legacy__development-guides__overview.html",            "legacy"),
    ("smarttv__develop__specifications__supported-standards.html",    "spec"),
    ("smarttv__develop__tools__ide__create-project.html",             "tool"),
    ("smarttv__develop__distribute__seller-office__submit.html",      "deployment"),
    ("smarttv__develop__samples__avplay-sample.html",                 "sample"),
    ("smarttv__develop__guides__user-interaction.html",               "guide"),
    ("smarttv__develop__getting-started__quick-start-guide.html",     "guide"),
    ("smarttv__develop__faq__common-issues.html",                     "faq"),
])
def test_samsung_tv_filenames(filename, expected):
    assert infer_doc_type(Path(filename)) == expected


# ---------------------------------------------------------------------------
# Directory-structured paths (future documentation sets)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path,expected", [
    ("docs/api-references/Button.html",          "api_reference"),
    ("docs/api/Text.html",                       "api_reference"),
    ("reference/components/View.html",           "api_reference"),
    ("docs/guides/getting-started.html",         "guide"),
    ("docs/tutorials/quickstart.html",           "guide"),
    ("docs/design/colors.html",                  "design"),
    ("docs/spec/config-schema.html",             "spec"),
    ("docs/tools/cli.html",                      "tool"),
    ("docs/samples/hello-world.html",            "sample"),
    ("docs/examples/counter.md",                 "sample"),
    ("docs/legacy/old-api.html",                 "legacy"),
    ("docs/deployment/publishing.html",          "deployment"),
    ("docs/faq/troubleshooting.html",            "faq"),
    ("docs/changelog/v2.html",                   "changelog"),
    ("docs/migration/v1-to-v2.html",             "migration"),
])
def test_directory_structured_paths(path, expected):
    assert infer_doc_type(Path(path)) == expected


# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------

def test_unknown_path_falls_back_to_guide():
    assert infer_doc_type(Path("docs/overview.html")) == "guide"


def test_empty_name_falls_back_to_guide():
    assert infer_doc_type(Path("page.html")) == "guide"
