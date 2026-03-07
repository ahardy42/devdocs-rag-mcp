#!/usr/bin/env python
"""CLI entry point for ingesting documentation into a collection."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from devdocs_rag.embedding import EmbeddingModel
from devdocs_rag.ingest.doc_type import infer_doc_type
from devdocs_rag.ingest.pipeline import SUPPORTED_EXTENSIONS, ingest
from devdocs_rag.store import DocStore


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest documentation into a ChromaDB collection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("path", help="Path to a file or directory of documentation")
    parser.add_argument("--collection", required=True, help="Collection name (e.g. samsung_tv)")
    parser.add_argument(
        "--doc-type",
        default=None,
        help="Doc type label applied to all ingested files",
    )
    parser.add_argument(
        "--infer-doc-type",
        action="store_true",
        help="Infer doc_type per file from path segments (api_reference, guide, design, etc.)",
    )
    parser.add_argument(
        "--drop",
        action="store_true",
        help="Drop the collection before ingesting (clean re-ingest)",
    )
    args = parser.parse_args()

    if args.doc_type and args.infer_doc_type:
        print("Error: --doc-type and --infer-doc-type are mutually exclusive", file=sys.stderr)
        sys.exit(1)

    embedding_model = EmbeddingModel()
    store = DocStore(embedding_model=embedding_model)

    if args.drop:
        if store.delete_collection(args.collection):
            print(f"Dropped collection '{args.collection}'")

    root = Path(args.path)

    if args.infer_doc_type:
        files = (
            [f for f in root.rglob("*") if f.suffix.lower() in SUPPORTED_EXTENSIONS]
            if root.is_dir()
            else [root]
        )
        total_files = total_chunks = 0
        all_errors: list[str] = []

        for file in files:
            result = ingest(
                file, args.collection, store, embedding_model,
                extra_metadata={"doc_type": infer_doc_type(file)},
            )
            total_files += result.files_processed
            total_chunks += result.chunks_created
            all_errors.extend(result.errors)

        print(f"Files processed : {total_files}")
        print(f"Chunks created  : {total_chunks}")
        if all_errors:
            print(f"Errors ({len(all_errors)}):", file=sys.stderr)
            for e in all_errors:
                print(f"  {e}", file=sys.stderr)
            sys.exit(1)

        stats = store.collection_stats(args.collection)
        print(f"Doc types       : {stats.doc_types}")
        return

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
