from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from f1_trivia_rag.rag.query_engine import load_query_engine

app = FastAPI(title="F1 Trivia RAG")

_query_engine = None


class ChatRequest(BaseModel):
    message: str


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
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    response = engine.query(request.message)
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
