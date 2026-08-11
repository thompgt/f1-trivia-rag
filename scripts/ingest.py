#!/usr/bin/env python
"""CLI entrypoint: fetch source data for the given seasons and rebuild the vector index.

Usage:
    python scripts/ingest.py --seasons 2021 2022 2023

Destructive: the Chroma collection is dropped and rebuilt, so the seasons passed here
become the entire corpus.
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from f1_trivia_rag.ingestion.pipeline import collect_documents
from f1_trivia_rag.rag.build_index import build_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", type=int, nargs="+", required=True)
    parser.add_argument(
        "--skip-wikipedia",
        action="store_true",
        help="Skip Wikipedia race-report ingestion (Ergast results only). Much faster.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Ignore the on-disk Ergast cache and re-fetch every request.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    documents = collect_documents(
        args.seasons,
        include_wikipedia=not args.skip_wikipedia,
        refresh=args.refresh,
    )

    print(f"Building index from {len(documents)} documents...")
    build_index(documents)
    print("Done.")


if __name__ == "__main__":
    main()
