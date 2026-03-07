#!/usr/bin/env python
"""CLI entry point for ingesting documentation into a collection."""

import argparse
import sys

from devdocs_rag.embedding import EmbeddingModel
from devdocs_rag.ingest.pipeline import ingest
from devdocs_rag.store import DocStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documentation into a ChromaDB collection.")
    parser.add_argument("path", help="Path to a file or directory of documentation")
    parser.add_argument("--collection", required=True, help="Collection name (e.g. samsung_tv)")
    parser.add_argument("--doc-type", default=None, help="Optional doc type label")
    args = parser.parse_args()

    embedding_model = EmbeddingModel()
    store = DocStore(embedding_model=embedding_model)
    extra = {"doc_type": args.doc_type} if args.doc_type else None

    result = ingest(args.path, args.collection, store, embedding_model, extra_metadata=extra)

    print(f"Files processed : {result.files_processed}")
    print(f"Chunks created  : {result.chunks_created}")
    print(f"Time (seconds)  : {result.time_seconds:.2f}")
    if result.errors:
        print(f"Errors ({len(result.errors)}):", file=sys.stderr)
        for e in result.errors:
            print(f"  {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
