"""The RAG chain: retrieve relevant chunks, rerank them, then generate a
grounded answer.

Retrieval/reranking (retrieve_context) and generation (build_chain) are
deliberately separate calls, not one combined RunnableParallel: retrieval
is fast and its result (the sources) is worth showing immediately, while
generation is the slow part worth streaming token by token. Uses
LangChain for the generation chain (matching AIAP Assignment 8's own
joined-context prompt shape), but with a hand-written prompt instead of
`hub.pull("rlm/rag-prompt")` - being able to show and explain your own
prompt text beats pointing at an anonymous hub artifact you didn't
author. The reranking step is a plain `sentence-transformers`
`CrossEncoder` call, not LangChain's own `CrossEncoderReranker` wrapper -
that class lives in the modern `langchain` umbrella package, which now
pulls in `langgraph` as a dependency even when unused, directly against
this project's own "no agent framework" stance.
"""

from functools import lru_cache

from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.vectorstores import VectorStoreRetriever
from sentence_transformers import CrossEncoder

from rag.config import MODEL, require_env

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
RERANK_TOP_N = 6

_PROMPT = ChatPromptTemplate.from_template(
    """Answer the question using only the information in the context below. \
Treat the context as the only source of truth - if it doesn't contain enough \
to answer, say so plainly rather than guessing or using outside knowledge.

Context:
{context}

Question: {question}

Answer in 2-4 short sentences, plain language, no unnecessary jargon."""
)


def build_retriever(
    vectorstore: Chroma, scheme: str | None = None, k: int = 20
) -> VectorStoreRetriever:
    """Builds a retriever, optionally scoped to one scheme's chunks.

    The gate already knows which scheme a question is about before
    retrieval runs, so retrieval doesn't need to rediscover that via an
    unscoped similarity search across the whole corpus - which risks
    pulling in a confusingly-similar chunk from a different scheme (e.g.
    Silver Support and ComCare both phrase income eligibility almost
    identically, so an unscoped search for one can surface the other).

    Args:
        vectorstore: The corpus vector store (see rag/ingest.py).
        scheme: If given, only retrieve chunks tagged with this scheme.
            None searches the whole corpus (used when the scheme isn't
            known yet, e.g. before the gate has run).
        k: How many candidate chunks the embedding search returns before
            reranking narrows them down to RERANK_TOP_N. Default 20, wider
            than the 6 actually used, since embedding similarity alone
            already proved unreliable at tightly ranking the right chunk
            (see the k=4->6 fix this module used to need on its own); the
            cross-encoder reranker in build_chain now does the precise
            ranking, this only needs to cast a wide enough net to make
            sure the right chunk is somewhere in the candidate set.

    Returns:
        A configured retriever.
    """
    search_kwargs: dict = {"k": k}
    if scheme is not None:
        search_kwargs["filter"] = {"scheme": scheme}
    return vectorstore.as_retriever(search_kwargs=search_kwargs)


@lru_cache(maxsize=1)
def _get_reranker() -> CrossEncoder:
    """Loads the cross-encoder once per process, not once per question."""
    return CrossEncoder(RERANK_MODEL)


def _rerank(
    question: str, docs: list[Document], top_n: int = RERANK_TOP_N
) -> list[Document]:
    """Re-scores a wider candidate set with a cross-encoder, keeps the best top_n.

    A cross-encoder reads the question and a chunk together and scores
    that specific pair directly, unlike the embedding similarity search
    that produced these candidates, which only ever compared two
    separately-computed vectors. More accurate, too slow to run over the
    whole corpus, so it narrows the retriever's wider candidate set rather
    than replacing the retriever.

    Args:
        question: The user's question.
        docs: Candidate chunks from the initial (wider) vector search.
        top_n: How many chunks to keep after reranking.

    Returns:
        The top_n docs, reordered by cross-encoder relevance score.
    """
    if not docs:
        return docs
    pairs = [(question, doc.page_content) for doc in docs]
    scores = _get_reranker().predict(pairs)
    ranked = sorted(zip(scores, docs), key=lambda pair: pair[0], reverse=True)
    return [doc for _, doc in ranked[:top_n]]


def _format_docs(docs: list[Document]) -> str:
    """Joins retrieved chunks into one context string for the prompt.

    Args:
        docs: The retrieved Document chunks.

    Returns:
        Their page_content joined with blank lines between chunks.
    """
    return "\n\n".join(doc.page_content for doc in docs)


def retrieve_context(retriever: VectorStoreRetriever, question: str) -> list[Document]:
    """Retrieves and reranks the chunks for a question, ahead of generation.

    Split out from build_chain so the caller can show sources (derived
    from this result via dedupe_sources) immediately, before the slower
    generation call streams in the answer.

    Args:
        retriever: A configured vectorstore retriever (see build_retriever).
        question: The user's question.

    Returns:
        The reranked chunks, ready for citation display or generation.
    """
    candidates = retriever.invoke(question)
    return _rerank(question, candidates)


def dedupe_sources(docs: list[Document]) -> list[tuple[str, str]]:
    """Deduplicates retrieved chunks down to one (scheme, source_url) pair per source.

    Args:
        docs: The retrieved Document chunks.

    Returns:
        (scheme, source_url) pairs, one per distinct URL, in first-seen
        order. Chunks with no source_url recorded are skipped.
    """
    seen: set[str] = set()
    sources: list[tuple[str, str]] = []
    for doc in docs:
        url = doc.metadata.get("source_url", "")
        if not url or url in seen:
            continue
        seen.add(url)
        sources.append((doc.metadata.get("scheme", ""), url))
    return sources


def build_chain() -> Runnable:
    """Builds the generation-only chain: reranked context + question -> answer.

    Retrieval happens separately via retrieve_context(); this chain just
    takes its result. Streamable: call `.stream({"context": ..., "question":
    ...})` instead of `.invoke(...)` to get the answer token by token.

    Returns:
        A runnable that takes a dict with `context` (a list of reranked
        Document chunks) and `question`, and returns the answer text.
    """
    require_env("ANTHROPIC_API_KEY")  # fail fast here, not mid-chain
    llm = ChatAnthropic(model=MODEL)

    return (
        (lambda x: {"context": _format_docs(x["context"]), "question": x["question"]})
        | _PROMPT
        | llm
        | StrOutputParser()
    )
