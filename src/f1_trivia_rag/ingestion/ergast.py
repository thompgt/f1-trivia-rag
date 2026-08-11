"""Fetches structured historical race data.

The original ergast.com/api was retired in 2024; api.jolpi.ca/ergast is the
community-run successor with the same schema/endpoints, so we default to it.

The API paginates everything and defaults to 30 rows a page, so every call here goes
through `_paginate`, which follows `offset`/`total` to the end. Requests retry with
exponential backoff (honouring `Retry-After`), and each season's raw JSON is cached on
disk so a failure part-way through a multi-season ingest does not throw away the
seasons already fetched.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator
from pathlib import Path

import requests

from f1_trivia_rag.config import settings
from f1_trivia_rag.ingestion.common import RawDocument

logger = logging.getLogger(__name__)

BASE_URL = "https://api.jolpi.ca/ergast/f1"

# The API caps `limit` at 100 and defaults to 30. The default is what silently truncated
# large grids: a 1950s race classifying more than 30 entrants lost the tail, mid-field.
PAGE_SIZE = 100

MAX_ATTEMPTS = 5
INITIAL_BACKOFF_SECONDS = 1.0
RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


class ErgastError(RuntimeError):
    """A request failed after exhausting retries."""


def _sleep(seconds: float) -> None:
    time.sleep(seconds)


def _retry_after(response: requests.Response, fallback: float) -> float:
    header = response.headers.get("Retry-After")
    if header:
        try:
            return max(float(header), 0.0)
        except ValueError:
            pass
    return fallback


def _request(path: str, params: dict) -> dict:
    """One GET, retried with exponential backoff on transient failures.

    Without this a single 429 part-way through a season aborted the whole ingest - and
    the public API rate-limits aggressively enough that this is the expected case, not
    an exotic one.
    """
    url = f"{BASE_URL}/{path}.json"
    backoff = INITIAL_BACKOFF_SECONDS
    last_error: Exception | None = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = requests.get(url, params=params, timeout=30)
            if response.status_code in RETRYABLE_STATUS:
                delay = _retry_after(response, backoff)
                last_error = ErgastError(f"{url} returned {response.status_code}")
                logger.warning(
                    "Ergast %s returned %s (attempt %d/%d); retrying in %.1fs",
                    path,
                    response.status_code,
                    attempt,
                    MAX_ATTEMPTS,
                    delay,
                )
            else:
                response.raise_for_status()
                return response.json()["MRData"]
        except requests.HTTPError as exc:
            # Any status reaching here is outside RETRYABLE_STATUS - a 404 for a season
            # that does not exist will not start existing on the third attempt.
            raise ErgastError(f"Ergast request failed: {path} ({exc})") from exc
        except requests.RequestException as exc:
            last_error = exc
            delay = backoff
            logger.warning(
                "Ergast %s failed (attempt %d/%d): %s; retrying in %.1fs",
                path,
                attempt,
                MAX_ATTEMPTS,
                exc,
                delay,
            )

        if attempt == MAX_ATTEMPTS:
            break
        _sleep(delay)
        backoff *= 2

    raise ErgastError(f"Ergast request failed after {MAX_ATTEMPTS} attempts: {path}") from last_error


def _paginate(path: str, *, sleep_between_requests: float = 0.2) -> Iterator[dict]:
    """Yields each page's MRData, following offset/total to the end of the collection."""
    offset = 0
    while True:
        data = _request(path, {"limit": PAGE_SIZE, "offset": offset})
        yield data

        total = int(data.get("total", 0))
        offset += PAGE_SIZE
        if offset >= total:
            return
        if sleep_between_requests:
            _sleep(sleep_between_requests)


def _cache_path(season: int, name: str) -> Path:
    return settings.data_raw_dir / "ergast" / str(season) / f"{name}.json"


def _cached(season: int, name: str, fetch, *, refresh: bool = False):
    """Reads `name` for `season` from the on-disk cache, or fetches and stores it.

    Resume support: a multi-season ingest that dies on round 14 of 2023 replays
    everything before it from disk instead of re-requesting it.
    """
    path = _cache_path(season, name)
    if not refresh and path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    value = fetch()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return value


def fetch_season_schedule(season: int, *, refresh: bool = False) -> list[dict]:
    def fetch() -> list[dict]:
        races: list[dict] = []
        for page in _paginate(f"{season}"):
            races.extend(page["RaceTable"]["Races"])
        return races

    return _cached(season, "schedule", fetch, refresh=refresh)


def fetch_race_result(season: int, round_: int, *, refresh: bool = False) -> dict | None:
    """One race with its full classification.

    `total` on this endpoint counts result *rows*, not races, so each page carries the
    same race with the next slice of `Results`; they are merged back into one race.
    """

    def fetch() -> dict | None:
        race: dict | None = None
        results: list[dict] = []
        for page in _paginate(f"{season}/{round_}/results"):
            races = page["RaceTable"]["Races"]
            if not races:
                break
            race = race or races[0]
            results.extend(races[0].get("Results", []))

        if race is None:
            return None
        return {**race, "Results": results}

    return _cached(season, f"round-{round_}", fetch, refresh=refresh)


def race_result_to_document(race: dict) -> RawDocument:
    season = race["season"]
    round_ = race["round"]
    race_name = race["raceName"]
    circuit = race["Circuit"]["circuitName"]
    date = race["date"]

    lines = [f"{race_name} ({season}), held at {circuit} on {date}."]
    for result in race.get("Results", []):
        driver = result["Driver"]
        constructor = result["Constructor"]["name"]
        position = result["position"]
        status = result["status"]
        lines.append(
            f"P{position}: {driver['givenName']} {driver['familyName']} "
            f"({constructor}) - {status}"
        )

    return RawDocument(
        text="\n".join(lines),
        source="ergast",
        source_id=f"{season}-{round_}-result",
        metadata={"season": season, "round": round_, "race_name": race_name},
    )


def fetch_season_results(
    season: int,
    *,
    sleep_between_requests: float = 0.2,
    refresh: bool = False,
) -> list[RawDocument]:
    """Fetches every race result for a season. One request per round (Ergast has no
    "all results for a season" endpoint that includes per-driver detail), so this is
    rate-limited with a small delay to stay under the public API's throttling.

    Results are cached per round, so re-running after a failure costs only the rounds
    that were never fetched.
    """
    schedule = fetch_season_schedule(season, refresh=refresh)
    documents = []
    for race in schedule:
        round_ = int(race["round"])
        result = fetch_race_result(season, round_, refresh=refresh)
        if result:
            documents.append(race_result_to_document(result))
        if sleep_between_requests:
            _sleep(sleep_between_requests)
    return documents
