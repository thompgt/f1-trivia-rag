#!/usr/bin/env python
"""CLI entrypoint: fetch source data for the given seasons and (re)build the vector index.

Usage:
    python scripts/ingest.py --seasons 2021 2022 2023
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from f1_trivia_rag.ingestion.ergast import fetch_season_results
from f1_trivia_rag.ingestion.common import RawDocument
from f1_trivia_rag.rag.build_index import build_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", type=int, nargs="+", required=True)
    parser.add_argument(
        "--skip-wikipedia",
        action="store_true",
        help="Skip Wikipedia race-report ingestion (Ergast results only).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    documents: list[RawDocument] = []
    for season in args.seasons:
        print(f"Fetching Ergast results for {season}...")
        documents.extend(fetch_season_results(season))

    if not args.skip_wikipedia:
        # TODO: derive grand_prix names per season from the Ergast schedule instead
        # of leaving this as a manual step.
        print("Skipping Wikipedia ingestion: wire up grand_prix names per season first.")

    print(f"Building index from {len(documents)} documents...")
    build_index(documents)
    print("Done.")


if __name__ == "__main__":
    main()
