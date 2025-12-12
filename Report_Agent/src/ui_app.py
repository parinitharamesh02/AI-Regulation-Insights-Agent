import sys
from pathlib import Path
from textwrap import shorten
from typing import List, Tuple

import streamlit as st

# -------------------------------------------------------------------
# Make project root importable so we can use `from src...` imports
# -------------------------------------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent  # folder that contains `src/`
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scraper import collect_articles
from src.chunking import semantic_chunk
from src.index import ChunkIndex
from src.llm_client import chat_completion, load_prompt, format_system_user
from src.config import TOPIC
from src.models import Chunk
from src.logging_utils import get_logger

logger = get_logger(__name__)

# -------------------------------------------------------------------
# Custom CSS — Professional dashboard look
# -------------------------------------------------------------------
CUSTOM_CSS = """
<style>
    .stApp {
        background-color: #020617;
        color: #e5e7eb;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1100px;
    }
    h1, h2, h3 {
        color: #f9fafb;
        font-weight: 600;
    }
    .sidebar .sidebar-content {
        background-color: #020617;
    }
    .stTextInput>div>div>input {
        background-color: #020617;
        color: #e5e7eb;
    }
    .stButton>button {
        background-color: #1d4ed8;
        color: #f9fafb;
        border-radius: 0.5rem;
        border: none;
        padding: 0.4rem 1.2rem;
        font-weight: 500;
    }
    .stButton>button:hover {
        background-color: #2563eb;
    }
    .chunk-metadata {
        font-size: 0.8rem;
        color: #9ca3af;
    }
    .chunk-preview {
        font-size: 0.9rem;
        color: #e5e7eb;
    }
</style>
"""


@st.cache_resource(show_spinner=True)
def build_knowledge_base() -> Tuple[ChunkIndex, List[Chunk]]:
    """
    Build the knowledge base once per session and cache it.

    Returns:
        (ChunkIndex, List[Chunk]): vector index and flat chunk list.
    """
    logger.info("Building knowledge base for UI...")
    articles = collect_articles()
    st.write(f"Fetched {len(articles)} articles.")

    all_chunks: List[Chunk] = []
    for art in articles:
        chunks = semantic_chunk(art)
        st.write(
            f"GOV.UK: '{shorten(art.title, width=60)}' — {len(chunks)} chunks"
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
    recent_turns = chat_history[-3:]

    history_str_parts: List[str] = []
    for turn in recent_turns:
        history_str_parts.append(f"User: {turn['question']}")
        history_str_parts.append(f"Assistant: {turn['answer']}")
    history_str = "\n".join(history_str_parts) if history_str_parts else "(no prior conversation)"

    context_parts: List[str] = []
    for i, (chunk, score) in enumerate(retrieved):
        context_parts.append(
            f"[CHUNK {i} | score={score:.3f} | article_id={chunk.article_id}]\n{chunk.text}\n"
        )
    context = "\n\n".join(context_parts) if context_parts else "(no retrieved context)"

    user_prompt = (
        f"Conversation so far:\n{history_str}\n\n"
        f"New user question:\n{question}\n\n"
        f"Retrieved context chunks:\n{context}\n\n"
        "Use the conversation history only to interpret what the user means. "
        "Base factual statements strictly on the retrieved chunks. "
        "If the context is insufficient, say that explicitly."
    )
    return user_prompt


def answer_question(
    question: str,
    index: ChunkIndex,
    chat_history: List[dict],
) -> Tuple[str, List[Tuple[Chunk, float]]]:
    """
    Retrieve relevant chunks and generate an answer using the Q&A prompt.
    """
    qa_prompt = load_prompt("qa_prompt")
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

    logger.info("UI answering question: %s", question)
    answer = chat_completion(messages)
    return answer, retrieved


def init_session_state() -> None:
    """
    Initialise Streamlit session state variables.
    """
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


def main() -> None:
    st.set_page_config(
        page_title="AI Regulation Insights Agent",
        layout="wide",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    init_session_state()

    st.title("AI Regulation Insights Agent")
    st.caption(f"Topic: {TOPIC}")

    col_left, col_right = st.columns([2, 1])

    with col_right:
        st.subheader("Knowledge Base")

        with st.expander("Build / Refresh Knowledge Base", expanded=True):
            rebuild = st.button("Rebuild Knowledge Base")
            if rebuild:
                build_knowledge_base.clear()
                st.success("Cache cleared. Knowledge base will rebuild on next question.")

            with st.spinner("Building knowledge base..."):
                index, chunks = build_knowledge_base()
            st.success("Knowledge base ready.")

        st.subheader("Recent Chunks (Preview)")
        # Show just a few previews
        for c in chunks[:5]:
            st.markdown(
                f"<div class='chunk-metadata'>[article_id={c.article_id}] "
                f"{shorten(c.text, width=80)}</div>",
                unsafe_allow_html=True,
            )

    with col_left:
        st.subheader("Ask a Question")

        question = st.text_input(
            "Enter your question",
            placeholder="e.g. What’s happening nowadays in UK AI regulation?",
        )

        if st.button("Submit") and question.strip():
            with st.spinner("Thinking..."):
                index, _ = build_knowledge_base()
                answer, retrieved = answer_question(
                    question=question.strip(),
                    index=index,
                    chat_history=st.session_state.chat_history,
                )

            st.markdown("**Assistant:**")
            st.write(answer)

            # Save to chat history
            st.session_state.chat_history.append(
                {
                    "question": question.strip(),
                    "answer": answer,
                    "sources": [c.article_id for c, _ in retrieved],
                }
            )

            with st.expander("Retrieved source chunks"):
                for i, (chunk, score) in enumerate(retrieved):
                    st.markdown(
                        f"<div class='chunk-metadata'>CHUNK {i} | score={score:.3f} | "
                        f"article_id={chunk.article_id}</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"<div class='chunk-preview'>{chunk.text}</div><hr/>",
                        unsafe_allow_html=True,
                    )

        if st.session_state.chat_history:
            st.subheader("Conversation History")
            for turn in reversed(st.session_state.chat_history):
                st.markdown(f"**You:** {turn['question']}")
                st.markdown(f"**Assistant:** {turn['answer']}")
                st.markdown("---")


if __name__ == "__main__":
    main()
