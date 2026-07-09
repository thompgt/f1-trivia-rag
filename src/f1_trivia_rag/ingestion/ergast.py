"""Fetches structured historical race data.

The original ergast.com/api was retired in 2024; api.jolpi.ca/ergast is the
community-run successor with the same schema/endpoints, so we default to it.
"""

import time

import requests

from f1_trivia_rag.ingestion.common import RawDocument

BASE_URL = "https://api.jolpi.ca/ergast/f1"


def _get(path: str) -> dict:
    resp = requests.get(f"{BASE_URL}/{path}.json", timeout=30)
    resp.raise_for_status()
    return resp.json()["MRData"]


def fetch_season_schedule(season: int) -> list[dict]:
    data = _get(f"{season}")
    return data["RaceTable"]["Races"]


def fetch_race_result(season: int, round_: int) -> dict | None:
    data = _get(f"{season}/{round_}/results")
    races = data["RaceTable"]["Races"]
    return races[0] if races else None


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


def fetch_season_results(season: int, *, sleep_between_requests: float = 0.2) -> list[RawDocument]:
    """Fetches every race result for a season. One request per round (Ergast has no
    "all results for a season" endpoint that includes per-driver detail), so this is
    rate-limited with a small delay to stay under the public API's throttling.
    """
    schedule = fetch_season_schedule(season)
    documents = []
    for race in schedule:
        round_ = int(race["round"])
        result = fetch_race_result(season, round_)
        if result:
            documents.append(race_result_to_document(result))
        time.sleep(sleep_between_requests)
    return documents
