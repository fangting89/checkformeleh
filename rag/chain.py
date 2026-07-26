"""The RAG chain: retrieve relevant chunks, then generate a grounded answer.

Uses LangChain for the retrieval mechanics (matching AIAP Assignment 8's
own RAG chain shape - RunnableParallel + a joined-context prompt), but
with a hand-written prompt instead of `hub.pull("rlm/rag-prompt")` -
being able to show and explain your own prompt text beats pointing at an
anonymous hub artifact you didn't author.
"""

from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableParallel, RunnablePassthrough
from langchain_core.vectorstores import VectorStoreRetriever

from rag.config import MODEL, require_env

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
    vectorstore: Chroma, scheme: str | None = None, k: int = 6
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
        k: How many chunks to retrieve. Default 6, not the more common 4 -
            two Day 3 eval failures were traced to the correct chunk
            ranking 5th, just outside a k=4 cutoff (see eval/run_eval.py's
            module docstring for how this was found).

    Returns:
        A configured retriever.
    """
    search_kwargs: dict = {"k": k}
    if scheme is not None:
        search_kwargs["filter"] = {"scheme": scheme}
    return vectorstore.as_retriever(search_kwargs=search_kwargs)


def _format_docs(docs: list[Document]) -> str:
    """Joins retrieved chunks into one context string for the prompt.

    Args:
        docs: The retrieved Document chunks.

    Returns:
        Their page_content joined with blank lines between chunks.
    """
    return "\n\n".join(doc.page_content for doc in docs)


def build_chain(retriever: VectorStoreRetriever) -> Runnable:
    """Builds the retrieve-then-generate chain.

    Args:
        retriever: A configured vectorstore retriever (see rag/ingest.py).

    Returns:
        A runnable that takes a question string and returns a dict with
        `context` (the retrieved Document objects, for citations),
        `question`, and `answer`.
    """
    require_env("ANTHROPIC_API_KEY")  # fail fast here, not mid-chain
    llm = ChatAnthropic(model=MODEL)

    return RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(
        answer=(
            lambda x: {"context": _format_docs(x["context"]), "question": x["question"]}
        )
        | _PROMPT
        | llm
        | StrOutputParser()
    )
