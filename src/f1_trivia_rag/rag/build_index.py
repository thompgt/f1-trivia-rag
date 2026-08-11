import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.errors import NotFoundError
from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from f1_trivia_rag.config import require_gemini_api_key, settings
from f1_trivia_rag.ingestion.common import RawDocument

# Chunking is pinned rather than inherited from whatever LlamaIndex's default happens
# to be in the installed version, because how documents split into nodes changes what
# a season-scoped retrieval has to fetch to cover a whole year. One Ergast race result
# fits comfortably inside one chunk; long Wikipedia reports split, and the retriever
# sizes itself from the resulting node count rather than assuming one node per race.
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 20


def _configure_llama_index() -> None:
    api_key = require_gemini_api_key()
    Settings.embed_model = GeminiEmbedding(
        model_name=settings.gemini_embed_model,
        api_key=api_key,
    )
    Settings.node_parser = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)


def to_llama_documents(raw_documents: list[RawDocument]) -> list[Document]:
    return [
        Document(
            text=doc.text,
            doc_id=f"{doc.source}:{doc.source_id}",
            metadata={"source": doc.source, "source_id": doc.source_id, **doc.metadata},
        )
        for doc in raw_documents
    ]


def reset_collection(chroma_client: ClientAPI, name: str) -> Collection:
    """Drops `name` if it exists and returns a fresh, empty collection.

    Indexing is a full rebuild, not an append. LlamaIndex mints a new random node id
    per run, so re-ingesting into an existing collection inserts a *second* copy of
    every race rather than replacing the first. That silently doubles the answer to
    exactly the aggregate questions this project exists to get right ("how many races
    did Red Bull win in 2023?"), so the collection is dropped first.
    """
    try:
        chroma_client.delete_collection(name)
    except (NotFoundError, ValueError):
        pass
    return chroma_client.create_collection(name)


def build_index(raw_documents: list[RawDocument]) -> VectorStoreIndex:
    """Rebuilds the persisted Chroma-backed vector index from raw documents.

    Destructive: any existing collection of the same name is dropped first, so the
    documents passed in are the entire corpus afterwards.
    """
    _configure_llama_index()

    settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(settings.chroma_persist_dir))
    chroma_collection = reset_collection(chroma_client, settings.chroma_collection)

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    llama_documents = to_llama_documents(raw_documents)
    index = VectorStoreIndex.from_documents(llama_documents, storage_context=storage_context)
    return index
