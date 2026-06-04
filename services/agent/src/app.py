"""
Legal RAG Chat UI — Streamlit frontend.
Run: streamlit run services/agent/src/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Make sure the src dir is on the path for local imports
_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import agent  # noqa: E402  (imported after path fix)

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Legal RAG",
    page_icon="⚖️",
    layout="wide",
)

# ─── Session state init ────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages: list[dict[str, str]] = []


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚖️ Legal RAG")
    st.caption("Indian Supreme Court Judgements")
    st.divider()

    if st.button("🔄 New Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption(f"Model: `{agent.LLM_MODEL}`")
    st.caption(f"Retrieval: `{agent.RETRIEVAL_URL}`")
    st.caption(f"top_k={agent.TOP_K} · rerank={agent.RERANK_TOP_N}")


# ─── Chat history display ──────────────────────────────────────────────────────
st.title("Legal Research Assistant")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ─── Input ────────────────────────────────────────────────────────────────────
if query := st.chat_input("Ask a legal question…"):

    # Display user message immediately
    st.chat_message("user").markdown(query)

    # Build sliding window of last 5 turns (passed to agent for context)
    history_window = st.session_state.messages[-10:]  # 5 turns = 10 msgs (user+assistant)

    # Append user message to state
    st.session_state.messages.append({"role": "user", "content": query})

    # Call agent
    with st.chat_message("assistant"):
        with st.spinner("Searching and thinking…"):
            try:
                result = agent.run(query, history_window)
                answer = result["answer"]
                chunks = result["chunks"]
                log_file = result["log_file"]
            except RuntimeError as exc:
                answer = f"⚠️ Error: {exc}"
                chunks = []
                log_file = ""

        st.markdown(answer)

        # Show source references in an expander
        if chunks:
            with st.expander(f"📄 Sources ({len(chunks)} chunks)", expanded=False):
                for chunk in chunks:
                    doc_id = chunk.get("doc_id", "unknown")
                    chunk_index = chunk.get("chunk_index", "?")
                    score = chunk.get("score") or chunk.get("rerank_score")
                    score_str = f" · score={score:.3f}" if score else ""
                    text_preview = (chunk.get("text") or "")[:200]
                    st.markdown(f"**`{doc_id}` · Chunk {chunk_index}{score_str}**")
                    st.caption(text_preview + ("…" if len(chunk.get("text", "")) > 200 else ""))
                    st.divider()

        if log_file:
            st.caption(f"📝 Log saved: `{log_file}`")

    # Append assistant response to state
    st.session_state.messages.append({"role": "assistant", "content": answer})
