"""Offline tests for FastF1 error handling, with fastf1 itself stubbed."""

import logging

import pytest
import requests
from fastf1.req import RateLimitExceededError

from f1_trivia_rag.ingestion import fastf1_source


@pytest.fixture(autouse=True)
def no_cache_setup(monkeypatch):
    monkeypatch.setattr(fastf1_source, "_ensure_cache", lambda: None)


def _raise_from_get_session(monkeypatch, error):
    def boom(season, grand_prix, session_type):
        raise error

    monkeypatch.setattr(fastf1_source.fastf1, "get_session", boom)


@pytest.mark.parametrize(
    "error",
    [
        ValueError("Invalid session type 'X'"),
        KeyError("EventName"),
        RateLimitExceededError("slow down"),
        requests.ConnectionError("no route to host"),
    ],
)
def test_unavailable_sessions_return_none_and_log(monkeypatch, caplog, error):
    _raise_from_get_session(monkeypatch, error)

    with caplog.at_level(logging.WARNING):
        result = fastf1_source.fetch_session_summary(2023, "Monaco Grand Prix")

    assert result is None
    assert "FastF1 has no" in caplog.text, "a skipped source must leave a trace"


def test_unexpected_errors_propagate(monkeypatch):
    """Regression: a bare `except Exception` turned a bug in this module into the same
    None a genuinely empty session returns, so a broken source looked like an empty one.
    """
    _raise_from_get_session(monkeypatch, AttributeError("typo in this module"))

    with pytest.raises(AttributeError):
        fastf1_source.fetch_session_summary(2023, "Monaco Grand Prix")
