import sys
from pathlib import Path
from textwrap import shorten
from typing import List, Tuple

import streamlit as st

# -------------------------------------------------------------------
# Make project root importable so we can use `from src...` imports
# -------------------------------------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent          # .../src/app
PROJECT_ROOT = CURRENT_DIR.parent.parent               # .../ai_insights_agent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TOPIC
from src.logging_utils import get_logger
from src.models import Chunk, Report
from src.scraping import collect_articles
from src.processing.chunking import semantic_chunk
from src.retrieval.index import ChunkIndex
from src.llm import chat_completion, load_prompt, format_system_user
from src.data.storage import (
    load_all_reports,
    save_chat_history,
    load_latest_chat_history,
)
from src.reporting.generate_report import generate_and_save_report
from src.reporting.trend_analysis import build_trend_analysis

logger = get_logger(__name__)

# -------------------------------------------------------------------
# Custom CSS — modern, clean, professional
# -------------------------------------------------------------------
CUSTOM_CSS = """
<style>
    .stApp {
        background: linear-gradient(135deg, #f9fafb 0%, #eef2ff 40%, #f9fafb 100%);
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
        color: #111827;
    }

    .main .block-container {
        padding-top: 2.5rem;
        padding-bottom: 3rem;
        max-width: 1100px;
        margin: 0 auto;
    }

    h1, h2, h3 {
        font-weight: 650;
        letter-spacing: -0.03em;
        color: #0f172a;
    }

    .section-subtitle {
        color: #6b7280;
        font-size: 0.9rem;
        margin-bottom: 1.5rem;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 1.4rem;
        border-bottom: 1px solid #e5e7eb;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 0.95rem;
        font-weight: 500;
        padding-bottom: 0.6rem;
    }

    .chat-card {
        border-radius: 16px;
        border: 1px solid #e5e7eb;
        background: #ffffff;
        padding: 1.25rem 1.5rem;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
        margin-bottom: 1.5rem;
    }

    .stChatMessage {
        max-width: 100%;
    }

    .stChatMessage:nth-child(odd) {
        background: #eff6ff;
        border-radius: 12px;
        padding: 0.5rem 0.75rem;
        margin-bottom: 0.4rem;
    }

    .stChatMessage:nth-child(even) {
        background: #f9fafb;
        border-radius: 12px;
        padding: 0.5rem 0.75rem;
        margin-bottom: 0.4rem;
        border-left: 3px solid #2563eb;
    }

    .stTextInput > div > div > input {
        border-radius: 999px;
        border: 1px solid #d1d5db;
        padding: 0.6rem 1rem;
        font-size: 0.95rem;
    }

    .stButton>button {
        border-radius: 999px;
        border: 1px solid #1d4ed8;
        padding: 0.4rem 1.2rem;
        font-size: 0.9rem;
        font-weight: 500;
        background-color: #2563eb;
        color: #ffffff;
    }
    .stButton>button:hover {
        background-color: #1d4ed8;
        border-color: #1d4ed8;
    }

    .metric-row {
        display: flex;
        gap: 1.5rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        flex: 1;
        border-radius: 14px;
        border: 1px solid #e5e7eb;
        background: #ffffff;
        padding: 0.9rem 1.1rem;
    }
    .metric-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #6b7280;
        margin-bottom: 0.15rem;
    }
    .metric-value {
        font-size: 1.4rem;
        font-weight: 600;
        color: #111827;
    }

    .chunk-box {
        border-radius: 10px;
        border: 1px solid #e5e7eb;
        background: #f9fafb;
        padding: 0.7rem 0.8rem;
        font-size: 0.85rem;
        color: #374151;
        margin-bottom: 0.6rem;
    }
</style>
"""


# -------------------------------------------------------------------
# Knowledge base construction
# -------------------------------------------------------------------
@st.cache_resource(show_spinner=True)
def build_knowledge_base() -> Tuple[ChunkIndex, List[Chunk]]:
    logger.info("Building knowledge base for UI...")
    articles = collect_articles()

    all_chunks: List[Chunk] = []
    for art in articles:
        chunks = semantic_chunk(art)
        logger.info(
            "Source=%s | title='%s' | chunks=%d",
            art.source,
            art.title[:80],
            len(chunks),
        )
        all_chunks.extend(chunks)

    index = ChunkIndex()
    index.build(all_chunks)
    logger.info("Knowledge base built with %d chunks", len(all_chunks))
    return index, all_chunks


def build_qa_prompt_with_history(
    question: str,
    retrieved: List[Tuple[Chunk, float]],
    chat_history: List[dict],
) -> str:
    """
    Build the user prompt including last N turns of conversation and retrieved chunks.
    """
    recent = chat_history[-3:]

    hist_parts: List[str] = []
    for turn in recent:
        hist_parts.append(f"User: {turn['question']}")
        hist_parts.append(f"Assistant: {turn['answer']}")
    history_str = "\n".join(hist_parts) if hist_parts else "(no prior conversation)"

    ctx_parts: List[str] = []
    for i, (chunk, score) in enumerate(retrieved):
        ctx_parts.append(
            f"[CHUNK {i} | score={score:.3f} | article_id={chunk.article_id}]\n{chunk.text}\n"
        )
    context = "\n\n".join(ctx_parts) if ctx_parts else "(no retrieved context)"

    user_prompt = (
        f"Conversation so far:\n{history_str}\n\n"
        f"New user question:\n{question}\n\n"
        f"Retrieved context chunks:\n{context}\n\n"
        "Use the conversation history only to interpret what the user means. "
        "Base factual statements strictly on the retrieved chunks. "
        "If the context is insufficient, say that explicitly."
    )
    return user_prompt


def is_trend_question(question: str) -> bool:
    """
    Heuristic routing: treat 'what changed', 'different from', 'trend', 'since last'
    as trend / change detection questions that should use the trend prompt.
    """
    q = question.lower()
    keywords = ["change", "changed", "different from", "trend", "evolving", "since last"]
    return any(k in q for k in keywords)


def answer_question(
    question: str,
    index: ChunkIndex,
    chat_history: List[dict],
) -> Tuple[str, List[Tuple[Chunk, float]]]:
    """
    Route between:
      - Trend analysis prompt (for 'change/trend' style questions)
      - Standard Q&A prompt over retrieved chunks
    """
    # Trend / change questions → use dedicated trend prompt over stored reports
    if is_trend_question(question):
        logger.info("Routing question to trend analysis: %s", question)
        trend = build_trend_analysis()
        if trend is not None:
            return trend, []
        # If not enough reports, fall back to normal Q&A

    # Normal RAG Q&A path
    qa_prompt = load_prompt("qa")
    retrieved = index.query(question, k=5)

    user_prompt = build_qa_prompt_with_history(
        question=question,
        retrieved=retrieved,
        chat_history=chat_history,
    )

    messages = format_system_user(
        system_prompt=qa_prompt,
        user_prompt=user_prompt,
    )

    logger.info("Routing question to standard Q&A: %s", question)
    answer = chat_completion(messages)
    return answer, retrieved


def init_session_state() -> None:
    """
    Initialise Streamlit session state variables, restoring latest chat history if available.
    """
    if "chat_history" not in st.session_state:
        # Load the most recent persisted chat if it exists
        history = load_latest_chat_history()
        st.session_state.chat_history = history or []

    if "last_retrieved" not in st.session_state:
        st.session_state.last_retrieved = []


# -------------------------------------------------------------------
# Main app
# -------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="AI Regulation Insights Agent",
        layout="wide",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    init_session_state()

    st.markdown("<h1>AI Regulation Insights Agent</h1>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='section-subtitle'>Topic: {TOPIC}. "
        "Answers are grounded in official GOV.UK policy and guidance documents.</div>",
        unsafe_allow_html=True,
    )

    tab_chat, tab_reports, tab_kb = st.tabs(["Chat", "Reports", "Knowledge Base"])

    # Ensure KB is ready once, shared across tabs
    with st.spinner("Preparing knowledge base..."):
        index, chunks = build_knowledge_base()

    # ----------------- Chat tab -----------------
    with tab_chat:
        st.markdown("<div class='chat-card'>", unsafe_allow_html=True)
        st.subheader("Conversational Q&A")
        st.markdown(
            "<div class='section-subtitle'>Ask specific questions or open prompts like "
            "“What’s happening nowadays in UK AI regulation?” or “How has the picture changed since last week?”.</div>",
            unsafe_allow_html=True,
        )

        # Existing conversation
        for turn in st.session_state.chat_history:
            with st.chat_message("user"):
                st.markdown(turn["question"])
            with st.chat_message("assistant"):
                st.markdown(turn["answer"])

        user_input = st.chat_input(
            "Ask a question about UK AI regulation and related policy developments"
        )

        if user_input is not None and user_input.strip():
            question = user_input.strip()

            with st.chat_message("user"):
                st.markdown(question)

            # Answer with resilience
            try:
                with st.chat_message("assistant"):
                    with st.spinner("Analysing relevant documents..."):
                        answer, retrieved = answer_question(
                            question=question,
                            index=index,
                            chat_history=st.session_state.chat_history,
                        )
                        st.markdown(answer)
            except Exception as e:
                logger.error("Error while answering question: %s", e)
                answer = (
                    "Sorry, something went wrong while generating an answer. "
                    "Please try again in a moment."
                )
                retrieved = []
                with st.chat_message("assistant"):
                    st.markdown(answer)

            # Update in-memory history
            st.session_state.chat_history.append(
                {
                    "question": question,
                    "answer": answer,
                    "sources": [c.article_id for c, _ in retrieved] if retrieved else [],
                }
            )
            st.session_state.last_retrieved = retrieved

            # Persist chat history (best-effort)
            try:
                save_chat_history(st.session_state.chat_history)
            except Exception as e:
                logger.warning("Failed to persist chat history: %s", e)

        st.markdown("</div>", unsafe_allow_html=True)

        if st.session_state.last_retrieved:
            with st.expander("Retrieved source chunks for the last answer"):
                for i, (chunk, score) in enumerate(st.session_state.last_retrieved):
                    st.markdown(
                        f"**CHUNK {i}** | score={score:.3f} | article_id={chunk.article_id}"
                    )
                    st.markdown(chunk.text)
                    st.markdown("---")

    # ----------------- Reports tab -----------------
    with tab_reports:
        st.subheader("Periodic Reports")
        st.markdown(
            "<div class='section-subtitle'>Generate short reports with key takeaways and extracted entities.</div>",
            unsafe_allow_html=True,
        )

        col_left, col_right = st.columns([2, 1])

        with col_left:
            if st.button("Generate report from latest knowledge base"):
                with st.spinner("Generating report..."):
                    report = generate_and_save_report(chunks)
                st.success(f"Report generated: {report.id}")

        reports = load_all_reports()

        with col_left:
            if reports:
                st.markdown("#### Report history (latest first)")
                for r in reversed(reports[-5:]):
                    with st.expander(f"{r.created_at.isoformat()}"):
                        st.markdown(f"**Summary**\n\n{r.summary}")
                        st.markdown("**Takeaways**")
                        for t in r.takeaways:
                            st.markdown(f"- {t}")
                        st.markdown("**Entities**")
                        for k, vals in r.entities.items():
                            if vals:
                                st.markdown(f"- **{k}**: {', '.join(sorted(set(vals)))}")
            else:
                st.info("No reports yet. Generate one using the button above.")

        with col_right:
            st.markdown("#### Latest report overview")
            if reports:
                latest: Report = reports[-1]
                st.markdown(f"**Created at:** {latest.created_at.isoformat()}")
                st.markdown(f"**Topic:** {latest.topic}")
                st.markdown("**Summary (preview)**")
                st.markdown(latest.summary)
            else:
                st.caption("No report generated yet.")

    # ----------------- Knowledge Base tab -----------------
    with tab_kb:
        st.subheader("Knowledge Base")
        st.markdown(
            "<div class='section-subtitle'>Overview of ingested articles and semantic chunks used for retrieval.</div>",
            unsafe_allow_html=True,
        )

        num_articles = len({c.article_id for c in chunks})
        num_chunks = len(chunks)

        st.markdown("<div class='metric-row'>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-label'>Articles ingested</div>
                <div class='metric-value'>{num_articles}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-label'>Semantic chunks</div>
                <div class='metric-value'>{num_chunks}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("#### Sample chunks")
        st.caption("Example chunks showing how the text is split semantically rather than by fixed size.")

        for c in chunks[:8]:
            st.markdown(f"**article_id**: `{c.article_id}` | **order**: {c.order}")
            st.markdown(shorten(c.text, width=230))
            st.markdown("---")


if __name__ == "__main__":
    main()
