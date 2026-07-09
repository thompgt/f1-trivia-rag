import chromadb
from llama_index.core import Document, StorageContext, VectorStoreIndex, Settings
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from f1_trivia_rag.config import settings
from f1_trivia_rag.ingestion.common import RawDocument


def _configure_llama_index() -> None:
    Settings.embed_model = GeminiEmbedding(
        model_name=settings.gemini_embed_model,
        api_key=settings.gemini_api_key,
    )


def to_llama_documents(raw_documents: list[RawDocument]) -> list[Document]:
    return [
        Document(
            text=doc.text,
            doc_id=f"{doc.source}:{doc.source_id}",
            metadata={"source": doc.source, "source_id": doc.source_id, **doc.metadata},
        )
        for doc in raw_documents
    ]


def build_index(raw_documents: list[RawDocument]) -> VectorStoreIndex:
    """Builds (or replaces) the persisted Chroma-backed vector index from raw documents."""
    _configure_llama_index()

    settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(settings.chroma_persist_dir))
    chroma_collection = chroma_client.get_or_create_collection(settings.chroma_collection)

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    llama_documents = to_llama_documents(raw_documents)
    index = VectorStoreIndex.from_documents(llama_documents, storage_context=storage_context)
    return index
