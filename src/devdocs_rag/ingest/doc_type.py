"""Doc type inference from file paths.

Extracts and tokenizes path segments from both directory-structured paths and
Samsung TV-style __ encoded filenames. Uses a scoring approach where:

  - Matching a full directory segment against a keyword scores 2 (strong match)
  - Matching a sub-token (split on hyphens/underscores) scores 1 (weak match)

The doc_type with the highest total score wins; ties break by rule order.
The filename (last segment) is excluded from matching to reduce false positives
from page names that happen to contain type-related words.

To extend for a new documentation set, add or update entries in _RULES. Include
compound forms (e.g. "api-references") for whole-segment matching alongside base
tokens (e.g. "api", "references") for sub-token matching.
"""

from __future__ import annotations

from pathlib import Path


# ---------------------------------------------------------------------------
# Rule table: (keywords, doc_type)
# Compound forms like "api-references" match whole segments (score 2).
# Base tokens like "api" match sub-tokens after splitting on - and _ (score 1).
# Higher total score wins; ties broken by rule order (first wins).
# ---------------------------------------------------------------------------
_RULES: list[tuple[set[str], str]] = [
    # API references
    ({"api-references", "api-reference", "api-refs", "apiref",
      "api", "reference", "references", "ref"}, "api_reference"),

    # Guides, tutorials, how-tos
    ({"guides", "guide", "tutorials", "tutorial", "howto", "how-to",
      "getting-started", "quickstart", "quick-start", "cookbook"}, "guide"),

    # Design / UX
    ({"design", "designs", "ux", "ui"}, "design"),

    # Specifications
    ({"specifications", "specification", "specs", "spec"}, "spec"),

    # Tools, SDK, CLI
    ({"tools", "tool", "sdk", "cli", "devtools", "ide"}, "tool"),

    # Samples, examples, demos
    ({"samples", "sample", "examples", "example", "demos", "demo",
      "playground", "snippets", "snippet"}, "sample"),

    # Legacy / deprecated
    ({"legacy", "deprecated", "archive"}, "legacy"),

    # Distribution / deployment / publishing
    ({"distribute", "distribution", "deployment", "deploy", "publish",
      "release", "submission"}, "deployment"),

    # FAQ
    ({"faq", "faqs", "troubleshooting", "troubleshoot"}, "faq"),

    # Changelogs / release notes
    ({"changelog", "changelogs", "release-notes", "whatsnew",
      "what-s-new", "history"}, "changelog"),

    # Migration
    ({"migration", "migrating", "upgrade", "upgrading"}, "migration"),
]

_FALLBACK = "guide"


def _path_segments(path: Path) -> tuple[set[str], set[str]]:
    """Return (whole_segs, sub_tokens_only) from the directory portion of path.

    whole_segs: normalised full segments (e.g. "api-references", "legacy")
    sub_tokens_only: tokens from splitting on - and _, excluding any token
                     that is itself a whole segment (avoids double-scoring).

    The final path segment (the filename / page name) is always excluded so
    that words in page titles don't corrupt structural classification.
    """
    parts = list(path.parts)

    if any("__" in p for p in parts):
        # Samsung TV __ encoding: flatten on __, then drop last (page name)
        all_segs: list[str] = []
        for p in parts:
            all_segs.extend(p.split("__"))
        dir_segs = all_segs[:-1]
    else:
        # Normal directory path: all parts except filename
        dir_segs = parts[:-1]

    whole: set[str] = set()
    for seg in dir_segs:
        seg = seg.lower().strip()
        if not seg:
            continue
        if "." in seg:
            seg = seg.rsplit(".", 1)[0]
        if seg:
            whole.add(seg)

    sub_only: set[str] = set()
    for seg in whole:
        for tok in seg.replace("_", "-").split("-"):
            tok = tok.strip()
            if tok and tok not in whole:
                sub_only.add(tok)

    return whole, sub_only


def infer_doc_type(path: Path) -> str:
    """Infer doc_type from a file path.

    Handles both directory-structured paths::

        docs/api-references/avplay.html  →  api_reference
        docs/guides/user-interaction.html  →  guide

    And Samsung TV-style __ encoded URL paths::

        smarttv__develop__api-references__avplay-api.html  →  api_reference
        smarttv__design__input-methods.html  →  design
    """
    whole, sub_only = _path_segments(path)

    best_type = _FALLBACK
    best_score = 0

    for keywords, doc_type in _RULES:
        score = 2 * len(keywords & whole) + len(keywords & sub_only)
        if score > best_score:
            best_score = score
            best_type = doc_type

    return best_type
