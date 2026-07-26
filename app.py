"""Streamlit UI: a single question box over the 6 supported schemes.

Fixed pipeline, no agent framework - the model never decides its own
steps: gate first, then (only if answerable) retrieve and generate. The
gate's own category decides which scheme's chunks get retrieved, so
retrieval never has to guess.

Run with: uv run streamlit run app.py
"""

import streamlit as st

from rag.chain import build_chain, build_retriever
from rag.gate import route_question
from rag.ingest import load_vectorstore

SCHEME_LABELS = {
    "cpf_life": "CPF LIFE",
    "silver_support": "Silver Support Scheme",
    "comcare": "ComCare",
    "lease_buyback": "HDB Lease Buyback Scheme",
    "ease": "EASE (Enhancement for Active Seniors)",
    "pioneer_merdeka": "Pioneer / Merdeka Generation Package",
}

st.set_page_config(page_title="AskLeh", page_icon="🧓")
st.title("AskLeh")
st.caption(
    "Ask about "
    + ", ".join(SCHEME_LABELS.values())
    + ". Anything else will be declined."
)


@st.cache_resource
def get_vectorstore():
    """Loads the vector store once per app process, not once per question.

    Only the vectorstore is cached, not the retrieval chain - the chain
    depends on which scheme the gate routes to, which varies per
    question, but building it is cheap local wiring (no embedding or
    API cost), so there's nothing to gain from caching it too.
    """
    return load_vectorstore()


question = st.text_input("Your question")

if question:
    with st.spinner("Checking..."):
        route = route_question(question)

    if route["decision"] == "decline":
        st.warning(
            "This assistant only covers: "
            + ", ".join(SCHEME_LABELS.values())
            + ". Please ask about one of these instead."
        )
    else:
        with st.spinner("Looking up an answer..."):
            retriever = build_retriever(get_vectorstore(), scheme=route["category"])
            chain = build_chain(retriever)
            result = chain.invoke(question)

        st.write(result["answer"])

        st.subheader("Sources")
        seen_urls: set[str] = set()
        for doc in result["context"]:
            url = doc.metadata.get("source_url", "")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            label = SCHEME_LABELS.get(doc.metadata.get("scheme"), "Source")
            st.markdown(f"- [{label}]({url})")
