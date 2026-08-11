import re

import chromadb
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.query_engine import CitationQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.vector_stores.types import FilterOperator, MetadataFilter, MetadataFilters
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.vector_stores.chroma import ChromaVectorStore

from f1_trivia_rag.config import settings

DEFAULT_TOP_K = 5
# A season has at most ~24 rounds; padded well above that so a season-scoped
# query always retrieves every race in that season instead of a similarity-ranked
# subset of it.
SEASON_TOP_K = 40
SEASON_PATTERN = re.compile(r"\b(19[5-9]\d|20\d{2})\b")


def _configure_llama_index() -> None:
    Settings.embed_model = GeminiEmbedding(
        model_name=settings.gemini_embed_model,
        api_key=settings.gemini_api_key,
    )
    Settings.llm = Gemini(model_name=settings.gemini_chat_model, api_key=settings.gemini_api_key)


class SeasonAwareRetriever(BaseRetriever):
    """Retrieves by plain similarity for lookup-style questions, but when the query
    names a specific season, switches to a season-filtered retrieval with a much
    higher top_k.

    Aggregate questions like "how many races did Red Bull win in 2023?" need every
    race from that season in context to be answered correctly. Similarity search alone
    only returns the `similarity_top_k` chunks closest to the query embedding - for a
    whole season that's a handful of the ~22 races, not all of them, so counts/sums the
    LLM computes over that partial context undercount (e.g. always landing on 5, the
    top_k itself, when every retrieved race happens to be a Red Bull win).
    """

    def __init__(self, index: VectorStoreIndex):
        self._index = index
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        match = SEASON_PATTERN.search(query_bundle.query_str)
        if match:
            filters = MetadataFilters(
                filters=[MetadataFilter(key="season", value=match.group(0), operator=FilterOperator.EQ)]
            )
            retriever = self._index.as_retriever(similarity_top_k=SEASON_TOP_K, filters=filters)
        else:
            retriever = self._index.as_retriever(similarity_top_k=DEFAULT_TOP_K)
        return retriever.retrieve(query_bundle)


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
    return CitationQueryEngine.from_args(index, retriever=SeasonAwareRetriever(index))
