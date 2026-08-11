"""Offline tests for the /chat endpoint, with the query engine stubbed."""

import pytest
from fastapi.testclient import TestClient
from google.genai import errors as genai_errors
from llama_index.core.base.response.schema import Response
from llama_index.core.schema import NodeWithScore, TextNode

import f1_trivia_rag.api.main as api_main
from f1_trivia_rag.config import MissingApiKeyError
from f1_trivia_rag.rag.query_engine import IndexUnavailableError


class StubEngine:
    def __init__(self, response=None, error=None):
        self._response = response
        self._error = error
        self.queries = []

    def query(self, message):
        self.queries.append(message)
        if self._error:
            raise self._error
        return self._response


def _answer(text="Verstappen [1].", source="ergast", source_id="2021-6-result"):
    node = NodeWithScore(
        node=TextNode(text="Monaco 2021", metadata={"source": source, "source_id": source_id}),
        score=1.0,
    )
    return Response(text, source_nodes=[node])


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api_main, "_query_engine", None)
    return TestClient(api_main.app, raise_server_exceptions=False)


def _install_engine(monkeypatch, engine):
    monkeypatch.setattr(api_main, "_get_query_engine", lambda: engine)
    return engine


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_successful_answer_carries_citations(client, monkeypatch):
    _install_engine(monkeypatch, StubEngine(response=_answer()))

    body = client.post("/chat", json={"message": "Who won Monaco 2021?"}).json()

    assert body["answer"] == "Verstappen [1]."
    assert body["citations"] == [{"source": "ergast", "source_id": "2021-6-result"}]


def test_empty_message_is_rejected(client):
    assert client.post("/chat", json={"message": ""}).status_code == 422


def test_overlong_message_is_rejected(client, monkeypatch):
    """Regression: message length was unbounded, so an arbitrarily large body went
    straight into a billed model call.
    """
    engine = _install_engine(monkeypatch, StubEngine(response=_answer()))

    response = client.post("/chat", json={"message": "x" * (api_main.MAX_MESSAGE_LENGTH + 1)})

    assert response.status_code == 422
    assert engine.queries == [], "rejected before reaching the model"


def test_message_at_the_limit_is_accepted(client, monkeypatch):
    _install_engine(monkeypatch, StubEngine(response=_answer()))

    response = client.post("/chat", json={"message": "x" * api_main.MAX_MESSAGE_LENGTH})

    assert response.status_code == 200


@pytest.mark.parametrize(
    "error",
    [
        IndexUnavailableError("collection is empty"),
        MissingApiKeyError("GEMINI_API_KEY is not set"),
    ],
)
def test_unavailable_index_or_config_is_503(client, monkeypatch, error):
    def boom():
        raise error

    monkeypatch.setattr(api_main, "_get_query_engine", boom)

    response = client.post("/chat", json={"message": "Who won Monaco 2021?"})

    assert response.status_code == 503


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        (429, 504),  # rate limited - upstream saying "later"
        (503, 504),
        (504, 504),
        (400, 502),
        (403, 502),
    ],
)
def test_upstream_model_errors_map_to_502_or_504(client, monkeypatch, code, expected):
    """Regression: only FileNotFoundError was caught, so any Gemini failure surfaced as
    an unhandled 500 - a server error attributed to the wrong server.
    """
    error = genai_errors.APIError(code, {"error": {"message": "upstream"}})
    _install_engine(monkeypatch, StubEngine(error=error))

    response = client.post("/chat", json={"message": "Who won Monaco 2021?"})

    assert response.status_code == expected
    assert str(code) in response.json()["detail"]


def test_timeouts_map_to_504(client, monkeypatch):
    _install_engine(monkeypatch, StubEngine(error=TimeoutError("too slow")))

    assert client.post("/chat", json={"message": "Who won?"}).status_code == 504
