import chromadb
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.query_engine import CitationQueryEngine
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore

from f1_trivia_rag.config import settings


def _configure_llama_index() -> None:
    Settings.embed_model = OpenAIEmbedding(
        model=settings.openai_embed_model,
        api_key=settings.openai_api_key,
    )
    Settings.llm = OpenAI(model=settings.openai_chat_model, api_key=settings.openai_api_key)


def load_query_engine() -> BaseQueryEngine:
    """Loads the persisted index and returns a citation-aware query engine, so answers
    reference which source document (Ergast result / Wikipedia report) backs each claim.
    """
    _configure_llama_index()

    if not settings.chroma_persist_dir.exists():
        raise FileNotFoundError(
            f"No index found at {settings.chroma_persist_dir}. Run scripts/ingest.py first."
        )

    chroma_client = chromadb.PersistentClient(path=str(settings.chroma_persist_dir))
    chroma_collection = chroma_client.get_or_create_collection(settings.chroma_collection)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    index = VectorStoreIndex.from_vector_store(vector_store)
    return CitationQueryEngine.from_args(index, similarity_top_k=5)
