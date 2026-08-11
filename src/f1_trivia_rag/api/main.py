import logging

from fastapi import FastAPI, HTTPException
from google.genai import errors as genai_errors
from pydantic import BaseModel, Field

from f1_trivia_rag.config import MissingApiKeyError
from f1_trivia_rag.rag.query_engine import IndexUnavailableError, load_query_engine

logger = logging.getLogger(__name__)

app = FastAPI(title="F1 Trivia RAG")

_query_engine = None

# A trivia question is a sentence. Without a bound, the request body is whatever the
# caller sends and it is forwarded straight into a billed model call.
MAX_MESSAGE_LENGTH = 2000

# Rate limits and timeouts are the upstream saying "later", not "no".
_UPSTREAM_RETRY_STATUS = frozenset({408, 429, 499, 500, 502, 503, 504})


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_LENGTH)


class Citation(BaseModel):
    source: str
    source_id: str


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]


def _get_query_engine():
    global _query_engine
    if _query_engine is None:
        _query_engine = load_query_engine()
    return _query_engine


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        engine = _get_query_engine()
    except (IndexUnavailableError, MissingApiKeyError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    try:
        response = engine.query(request.message)
    except genai_errors.APIError as exc:
        # An upstream failure is not this service's failure: 502 for a bad response,
        # 504 when the model timed out or rate-limited us. Previously any of these
        # escaped as an unhandled 500.
        status = 504 if exc.code in _UPSTREAM_RETRY_STATUS else 502
        logger.warning("Gemini call failed with %s: %s", exc.code, exc)
        raise HTTPException(
            status_code=status, detail=f"Upstream model error ({exc.code})."
        ) from exc
    except TimeoutError as exc:
        logger.warning("Gemini call timed out: %s", exc)
        raise HTTPException(status_code=504, detail="Upstream model timed out.") from exc

    citations = [
        Citation(
            source=node.metadata.get("source", "unknown"),
            source_id=node.metadata.get("source_id", "unknown"),
        )
        for node in response.source_nodes
    ]
    return ChatResponse(answer=str(response), citations=citations)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
