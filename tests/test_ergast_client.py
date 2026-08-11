"""Offline tests for the Ergast HTTP client: pagination, retries, and the cache.

No network: `requests.get` is replaced with a stub that serves canned pages and
records the query parameters it was called with.
"""

import json

import pytest
import requests

from f1_trivia_rag.ingestion import ergast


class FakeResponse:
    def __init__(self, payload=None, status_code=200, headers=None):
        self._payload = payload or {}
        self.status_code = status_code
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")

    def json(self):
        return self._payload


def _mrdata(total, races):
    return {"MRData": {"total": str(total), "RaceTable": {"Races": races}}}


def _race(round_, results):
    return {
        "season": "2023",
        "round": str(round_),
        "raceName": f"Race {round_}",
        "date": "2023-03-05",
        "Circuit": {"circuitName": "Some Circuit"},
        "Results": results,
    }


def _result_row(position):
    return {
        "position": str(position),
        "status": "Finished",
        "Driver": {"givenName": "Driver", "familyName": str(position)},
        "Constructor": {"name": "Some Team"},
    }


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(ergast.settings, "data_raw_dir", tmp_path)
    monkeypatch.setattr(ergast, "_sleep", lambda _seconds: None)


def _install(monkeypatch, pages):
    """Serves `pages` in order and records each call's params."""
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append({"url": url, "params": params})
        return pages[len(calls) - 1]

    monkeypatch.setattr(requests, "get", fake_get)
    return calls


def test_schedule_follows_pagination_past_the_default_page(monkeypatch):
    """Regression: limit/offset were never sent, so the API's 30-row default silently
    truncated anything longer.
    """
    first = [_race(i, []) for i in range(1, 101)]
    second = [_race(i, []) for i in range(101, 123)]
    calls = _install(
        monkeypatch,
        [FakeResponse(_mrdata(122, first)), FakeResponse(_mrdata(122, second))],
    )

    races = ergast.fetch_season_schedule(2023)

    assert len(races) == 122
    assert [c["params"] for c in calls] == [
        {"limit": 100, "offset": 0},
        {"limit": 100, "offset": 100},
    ]


def test_single_page_makes_a_single_request(monkeypatch):
    calls = _install(monkeypatch, [FakeResponse(_mrdata(3, [_race(i, []) for i in (1, 2, 3)]))])

    assert len(ergast.fetch_season_schedule(2023)) == 3
    assert len(calls) == 1


def test_race_results_are_merged_across_pages(monkeypatch):
    """`total` on the results endpoint counts result rows, so a large grid arrives as
    the same race repeated with successive slices of Results.
    """
    page_one = [_race(1, [_result_row(p) for p in range(1, 101)])]
    page_two = [_race(1, [_result_row(p) for p in range(101, 108)])]
    _install(
        monkeypatch,
        [FakeResponse(_mrdata(107, page_one)), FakeResponse(_mrdata(107, page_two))],
    )

    race = ergast.fetch_race_result(2023, 1)

    assert len(race["Results"]) == 107
    assert race["Results"][-1]["position"] == "107"


def test_missing_race_returns_none(monkeypatch):
    _install(monkeypatch, [FakeResponse(_mrdata(0, []))])

    assert ergast.fetch_race_result(2023, 99) is None


def test_a_429_is_retried_rather_than_aborting_the_ingest(monkeypatch):
    """Regression: one 429 used to discard every season fetched so far."""
    calls = _install(
        monkeypatch,
        [
            FakeResponse(status_code=429, headers={"Retry-After": "0"}),
            FakeResponse(_mrdata(1, [_race(1, [])])),
        ],
    )

    assert len(ergast.fetch_season_schedule(2023)) == 1
    assert len(calls) == 2


def test_connection_errors_are_retried(monkeypatch):
    calls = []

    def flaky_get(url, params=None, timeout=None):
        calls.append(params)
        if len(calls) == 1:
            raise requests.ConnectionError("boom")
        return FakeResponse(_mrdata(1, [_race(1, [])]))

    monkeypatch.setattr(requests, "get", flaky_get)

    assert len(ergast.fetch_season_schedule(2023)) == 1
    assert len(calls) == 2


def test_retries_are_bounded_and_then_raise(monkeypatch):
    _install(monkeypatch, [FakeResponse(status_code=503)] * ergast.MAX_ATTEMPTS)

    with pytest.raises(ergast.ErgastError):
        ergast.fetch_season_schedule(2023)


def test_a_non_retryable_error_is_not_retried(monkeypatch):
    calls = _install(monkeypatch, [FakeResponse(status_code=404)])

    with pytest.raises(ergast.ErgastError):
        ergast.fetch_season_schedule(2023)

    assert len(calls) == 1


def test_responses_are_cached_on_disk_so_a_rerun_resumes(monkeypatch, tmp_path):
    calls = _install(monkeypatch, [FakeResponse(_mrdata(1, [_race(1, [])]))])

    first = ergast.fetch_season_schedule(2023)
    second = ergast.fetch_season_schedule(2023)

    assert first == second
    assert len(calls) == 1, "second call must be served from the cache"
    assert json.loads((tmp_path / "ergast" / "2023" / "schedule.json").read_text()) == first


def test_refresh_bypasses_the_cache(monkeypatch):
    calls = _install(
        monkeypatch,
        [FakeResponse(_mrdata(1, [_race(1, [])])), FakeResponse(_mrdata(1, [_race(2, [])]))],
    )

    ergast.fetch_season_schedule(2023)
    refreshed = ergast.fetch_season_schedule(2023, refresh=True)

    assert len(calls) == 2
    assert refreshed[0]["round"] == "2"
