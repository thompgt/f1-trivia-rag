"""Assembles the corpus for a set of seasons from the individual sources."""

import logging
from collections.abc import Iterable

from f1_trivia_rag.ingestion.common import RawDocument
from f1_trivia_rag.ingestion.ergast import fetch_season_results, fetch_season_schedule
from f1_trivia_rag.ingestion.wikipedia_source import fetch_race_reports

logger = logging.getLogger(__name__)


def grand_prix_names(season: int, *, refresh: bool = False) -> list[str]:
    """The Grand Prix names of a season, in calendar order, de-duplicated.

    Ergast's `raceName` ("Monaco Grand Prix") is already the Wikipedia title fragment
    the report fetcher wants, so the schedule the ingest fetches anyway is enough to
    drive Wikipedia ingestion - it never needed a hand-maintained list.
    """
    schedule = fetch_season_schedule(season, refresh=refresh)
    return list(dict.fromkeys(race["raceName"] for race in schedule))


def collect_documents(
    seasons: Iterable[int],
    *,
    include_wikipedia: bool = True,
    refresh: bool = False,
) -> list[RawDocument]:
    """Fetches every source document for `seasons`."""
    documents: list[RawDocument] = []

    for season in seasons:
        logger.info("Fetching Ergast results for %s...", season)
        documents.extend(fetch_season_results(season, refresh=refresh))

        if not include_wikipedia:
            continue

        names = grand_prix_names(season, refresh=refresh)
        logger.info("Fetching %d Wikipedia race reports for %s...", len(names), season)
        reports = fetch_race_reports(season, names)
        if len(reports) < len(names):
            logger.warning(
                "Wikipedia returned %d of %d race reports for %s (missing or ambiguous titles).",
                len(reports),
                len(names),
                season,
            )
        documents.extend(reports)

    return documents
