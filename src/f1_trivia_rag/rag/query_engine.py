import logging
import re

import chromadb
from chromadb.api.models.Collection import Collection
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.response.schema import RESPONSE_TYPE, Response
from llama_index.core.prompts import PromptTemplate
from llama_index.core.query_engine import CitationQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.vector_stores.types import FilterOperator, MetadataFilter, MetadataFilters
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.vector_stores.chroma import ChromaVectorStore

from f1_trivia_rag.config import settings

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 5

# Season-scoped retrieval sizes itself from the number of nodes actually stored for
# that season rather than from a guessed constant. The old fixed cap of 40 assumed
# "~24 rounds, one node per race", but nothing enforces one node per race: the node
# parser splits long documents (a Wikipedia race report becomes many nodes), so a
# single season can hold far more than 40 nodes and the retrieval would silently
# return a subset - undercounting exactly the aggregate questions the filter exists
# to get right. This constant is only a safety ceiling so a pathological corpus
# cannot pull an unbounded number of nodes into one LLM context.
MAX_SEASON_TOP_K = 400

SEASON_PATTERN = re.compile(r"\b(19[5-9]\d|20\d{2})\b")

# Returned without calling the LLM when retrieval finds nothing - typically a question
# about a season that was never ingested. LlamaIndex's own empty-node path returns the
# opaque string "Empty Response"; this says what actually happened.
NO_SOURCES_ANSWER = (
    "I could not find anything about that in the indexed sources, so I cannot answer it. "
    "The index only covers the seasons that have been ingested - run scripts/ingest.py "
    "for the season you are asking about."
)

_GROUNDING_RULES = (
    "You answer Formula 1 questions using ONLY the numbered sources below.\n"
    "Rules:\n"
    "1. Never use prior knowledge about Formula 1. If a fact is not written in the "
    "sources, you do not know it - even if you are confident it is true.\n"
    "2. Cite the source number(s) each claim comes from, like [1] or [2][5].\n"
    "3. If the sources do not answer the question, say so plainly and cite nothing. "
    "Do not guess, and do not attach a citation to a fact the cited source does not "
    "state - a wrong citation is worse than no answer.\n"
    "4. For counting or 'which races did X not win' questions, count only over the "
    "sources shown. Answering zero is correct when the sources show none; say what you "
    "counted over.\n"
)

_EXAMPLE = (
    "Example:\n"
    "Source 1:\n"
    "Monaco Grand Prix (2021). P1: Max Verstappen (Red Bull) - Finished\n"
    "Source 2:\n"
    "Azerbaijan Grand Prix (2021). P1: Sergio Perez (Red Bull) - Finished\n"
    "Query: How many races did Red Bull win?\n"
    "Answer: Two - Monaco [1] and Azerbaijan [2].\n"
)

# The stock CitationQueryEngine prompt only mildly suggests abstention ("if none of the
# sources are helpful, you should indicate that"), which leaves the model free to answer
# an out-of-corpus question from parametric memory and hang a citation off an unrelated
# retrieved race. For a project whose whole claim is "checkable answers", grounding has
# to be an instruction, not a hint.
F1_CITATION_QA_TEMPLATE = PromptTemplate(
    _GROUNDING_RULES
    + _EXAMPLE
    + "\nNow it's your turn. Below are several numbered sources of information:"
    "\n------\n"
    "{context_str}"
    "\n------\n"
    "Query: {query_str}\n"
    "Answer: "
)

F1_CITATION_REFINE_TEMPLATE = PromptTemplate(
    _GROUNDING_RULES
    + _EXAMPLE
    + "\nNow it's your turn. We have provided an existing answer: {existing_answer}\n"
    "Below are more numbered sources. Use them to refine the existing answer, keeping "
    "every rule above. If they add nothing, repeat the existing answer unchanged."
    "\n------\n"
    "{context_msg}"
    "\n------\n"
    "Query: {query_str}\n"
    "Answer: "
)


def _configure_llama_index() -> None:
    Settings.embed_model = GeminiEmbedding(
        model_name=settings.gemini_embed_model,
        api_key=settings.gemini_api_key,
    )
    Settings.llm = Gemini(model_name=settings.gemini_chat_model, api_key=settings.gemini_api_key)


class SeasonAwareRetriever(BaseRetriever):
    """Retrieves by plain similarity for lookup-style questions, but when the query
    names a specific season, switches to a season-filtered retrieval sized to that
    season's entire stored contents.

    Aggregate questions like "how many races did Red Bull win in 2023?" need every
    race from that season in context to be answered correctly. Similarity search alone
    only returns the `similarity_top_k` chunks closest to the query embedding - for a
    whole season that's a handful of the ~22 races, not all of them, so counts/sums the
    LLM computes over that partial context undercount (e.g. always landing on 5, the
    top_k itself, when every retrieved race happens to be a Red Bull win).

    The top_k is read from the store rather than assumed: the retriever counts the
    nodes carrying that season and asks for exactly that many (up to
    `MAX_SEASON_TOP_K`), so the "sees the whole season" guarantee survives a node
    parser that splits one race into several nodes.
    """

    def __init__(self, index: VectorStoreIndex, collection: Collection):
        self._index = index
        self._collection = collection
        super().__init__()

    def _season_node_count(self, season: str) -> int:
        """How many nodes are stored under this season - the number a season-scoped
        retrieval has to return to genuinely cover the whole year.
        """
        return len(self._collection.get(where={"season": season}, include=[])["ids"])

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        match = SEASON_PATTERN.search(query_bundle.query_str)
        if not match:
            return self._index.as_retriever(similarity_top_k=DEFAULT_TOP_K).retrieve(query_bundle)

        season = match.group(0)
        stored = self._season_node_count(season)
        top_k = min(max(stored, DEFAULT_TOP_K), MAX_SEASON_TOP_K)
        if stored > MAX_SEASON_TOP_K:
            logger.warning(
                "Season %s holds %d nodes, above the %d ceiling; the answer will be based "
                "on a subset and aggregates over this season may undercount.",
                season,
                stored,
                MAX_SEASON_TOP_K,
            )

        filters = MetadataFilters(
            filters=[MetadataFilter(key="season", value=season, operator=FilterOperator.EQ)]
        )
        nodes = self._index.as_retriever(similarity_top_k=top_k, filters=filters).retrieve(query_bundle)

        if len(nodes) < min(stored, MAX_SEASON_TOP_K):
            logger.warning(
                "Season %s: retrieved %d of %d stored nodes; aggregates may undercount.",
                season,
                len(nodes),
                stored,
            )
        return nodes


class AbstainingCitationQueryEngine(CitationQueryEngine):
    """A CitationQueryEngine that refuses explicitly when retrieval came back empty.

    A question about an un-ingested season filters down to zero nodes. LlamaIndex's
    synthesizer does skip the LLM in that case, but it answers the literal string
    "Empty Response", which the API would happily serve as a 200 alongside an empty
    citation list. Replacing it with a real explanation keeps "no answer" visibly
    distinct from "an answer with no sources".
    """

    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        return self._guard(super()._query(query_bundle))

    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        return self._guard(await super()._aquery(query_bundle))

    @staticmethod
    def _guard(response: RESPONSE_TYPE) -> RESPONSE_TYPE:
        if not response.source_nodes:
            return Response(response=NO_SOURCES_ANSWER, source_nodes=[])
        return response


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
    retriever = SeasonAwareRetriever(index, chroma_collection)
    return AbstainingCitationQueryEngine.from_args(
        index,
        retriever=retriever,
        citation_qa_template=F1_CITATION_QA_TEMPLATE,
        citation_refine_template=F1_CITATION_REFINE_TEMPLATE,
    )
