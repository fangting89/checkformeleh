"""Streamlit UI: a single question box over the 6 supported schemes.

Fixed pipeline, no agent framework - the model never decides its own
steps: gate first, then (only if answerable) retrieve, rerank, and
generate. The gate's own category decides which scheme's chunks get
retrieved, so retrieval never has to guess. Retrieval resolves first, so
sources render immediately, then the answer streams in token by token.

Rate-capped per session (MAX_QUESTIONS_PER_SESSION below) as a cost/abuse
control, same reasoning and same scope limitation as readformeleh's demo:
per-session only, not a persisted daily cap, since a static Streamlit
Community Cloud demo has no shared store to count against across
sessions/restarts.

Run with: uv run streamlit run app.py
"""

import streamlit as st

from rag.chain import build_chain, build_retriever, dedupe_sources, retrieve_context
from rag.gate import route_question
from rag.ingest import load_vectorstore

MAX_QUESTIONS_PER_SESSION = 30

SCHEME_LABELS = {
    "cpf_life": "CPF LIFE",
    "silver_support": "Silver Support Scheme",
    "comcare": "ComCare",
    "lease_buyback": "HDB Lease Buyback Scheme",
    "ease": "EASE (Enhancement for Active Seniors)",
    "pioneer_merdeka": "Pioneer / Merdeka Generation Package",
}

EXAMPLE_QUESTIONS = [
    ("CPF LIFE", "What is the minimum retirement savings needed to join CPF LIFE?"),
    ("Silver Support", "How much does Silver Support pay each quarter?"),
    ("ComCare", "What income qualifies someone for ComCare assistance?"),
    ("Lease Buyback", "What is the Lease Buyback Scheme bonus for a 3-room flat?"),
    ("EASE", "What percentage of the EASE improvement cost does the government pay?"),
    (
        "Pioneer/Merdeka",
        "What year must someone be born to qualify for the Pioneer Generation Package?",
    ),
]

# The theme deliberately echoes Singapore's real Government Design System
# (see .streamlit/config.toml), so this disclaimer isn't optional polish,
# it's what keeps a convincingly official-looking demo from being mistaken
# for an actual government channel.
DISCLAIMER = (
    "Independent personal portfolio project. Not affiliated with, "
    "endorsed by, or an official channel of the Singapore Government."
)

st.set_page_config(page_title="CheckForMeLeh", page_icon="🧓")

with st.sidebar:
    st.header("About CheckForMeLeh")
    st.write(
        "Answers questions about 6 Singapore senior support schemes from "
        "a small, hand-verified corpus of real government pages and "
        "brochures."
    )
    st.subheader("Covered schemes")
    for label in SCHEME_LABELS.values():
        st.write(f"- {label}")
    st.divider()
    st.caption(DISCLAIMER)
    st.caption("[Source on GitHub](https://github.com/fangting89/checkformeleh)")

st.title("Senior Support Scheme Q&A Assistant")
st.caption("checkformeleh")
st.caption(
    "Ask a question about Singapore senior support schemes, answered in plain language."
)
st.caption(f"⚠️ {DISCLAIMER}")


@st.cache_resource
def get_vectorstore():
    """Loads the vector store once per app process, not once per question.

    Only the vectorstore is cached, not the retrieval chain - the chain
    depends on which scheme the gate routes to, which varies per
    question, but building it is cheap local wiring (no embedding or
    API cost), so there's nothing to gain from caching it too.
    """
    return load_vectorstore()


def _set_question(text: str) -> None:
    st.session_state.question = text


st.write("Try an example:")
EXAMPLES_PER_ROW = 3
for row_start in range(0, len(EXAMPLE_QUESTIONS), EXAMPLES_PER_ROW):
    row = EXAMPLE_QUESTIONS[row_start : row_start + EXAMPLES_PER_ROW]
    for col, (label, text) in zip(st.columns(EXAMPLES_PER_ROW), row):
        col.button(label, on_click=_set_question, args=(text,))

question = st.text_input("Your question", key="question")

questions_asked = st.session_state.get("questions_asked", 0)
if question and questions_asked >= MAX_QUESTIONS_PER_SESSION:
    st.warning(
        f"⚠️ This demo caps questions at {MAX_QUESTIONS_PER_SESSION} per session "
        "to control API cost. Refresh the page to reset."
    )
elif question:
    st.session_state.questions_asked = questions_asked + 1
    with st.spinner("Checking..."):
        route = route_question(question)

    if route["decision"] == "decline":
        st.warning(
            "This assistant only covers: "
            + ", ".join(SCHEME_LABELS.values())
            + ". Please ask about one of these instead."
        )
    else:
        with st.spinner("Looking up sources..."):
            retriever = build_retriever(get_vectorstore(), scheme=route["category"])
            context = retrieve_context(retriever, question)

        # Sources are already fully resolved at this point (retrieval,
        # unlike generation, isn't worth streaming), so show them before
        # the answer streams in below, rather than making the user wait
        # for the whole answer just to see where it came from.
        sources = dedupe_sources(context)
        if sources:
            with st.expander(f"Sources ({len(sources)})", expanded=True):
                for scheme, url in sources:
                    label = SCHEME_LABELS.get(scheme, "Source")
                    st.markdown(f"- [{label}]({url})")

        with st.container(border=True):
            st.markdown("**Answer**")
            chain = build_chain()
            st.write_stream(chain.stream({"context": context, "question": question}))
