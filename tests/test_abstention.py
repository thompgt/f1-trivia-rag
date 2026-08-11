"""Offline tests for the empty-retrieval guard and the grounding prompts."""

from llama_index.core.base.response.schema import Response
from llama_index.core.schema import NodeWithScore, TextNode

from f1_trivia_rag.rag.query_engine import (
    F1_CITATION_QA_TEMPLATE,
    F1_CITATION_REFINE_TEMPLATE,
    NO_SOURCES_ANSWER,
    AbstainingCitationQueryEngine,
)


def test_empty_retrieval_is_replaced_with_an_explicit_refusal():
    """LlamaIndex answers the opaque literal "Empty Response" when nothing was
    retrieved. Served through /chat that reads as an answer with no citations.
    """
    guarded = AbstainingCitationQueryEngine._guard(Response("Empty Response", source_nodes=[]))

    assert guarded.response == NO_SOURCES_ANSWER
    assert guarded.source_nodes == []


def test_a_real_answer_passes_through_untouched():
    nodes = [NodeWithScore(node=TextNode(text="Monaco 2021. P1: Max Verstappen"), score=1.0)]
    original = Response("Verstappen [1].", source_nodes=nodes)

    assert AbstainingCitationQueryEngine._guard(original) is original


def test_refusal_text_names_the_remedy():
    assert "ingest" in NO_SOURCES_ANSWER


def test_qa_template_forbids_prior_knowledge_and_requires_citations():
    prompt = F1_CITATION_QA_TEMPLATE.template.lower()

    assert "only the numbered sources" in prompt
    assert "never use prior knowledge" in prompt
    assert "do not guess" in prompt
    # Answering zero must stay allowed, or season aggregates regress into refusals.
    assert "answering zero is correct" in prompt


def test_templates_declare_the_variables_llamaindex_will_fill():
    assert "{context_str}" in F1_CITATION_QA_TEMPLATE.template
    assert "{query_str}" in F1_CITATION_QA_TEMPLATE.template
    assert "{context_msg}" in F1_CITATION_REFINE_TEMPLATE.template
    assert "{existing_answer}" in F1_CITATION_REFINE_TEMPLATE.template
    assert "{query_str}" in F1_CITATION_REFINE_TEMPLATE.template
