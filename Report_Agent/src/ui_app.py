import sys
from pathlib import Path
from textwrap import shorten
from typing import List

import streamlit as st

# -------------------------------------------------------------------
# Make project root importable so we can use `from src...` imports
# -------------------------------------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent  # folder that contains `src/`

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Now we can safely do absolute imports
from src.scraper import collect_articles
from src.chunking import semantic_chunk
from src.index import ChunkIndex
from src.llm_client import chat_completion, load_prompt, format_system_user
from src.config import TOPIC

# -------------------------------------------------------------------
# Custom CSS — Professional dashboard look
# -------------------------------------------------------------------

CUSTOM_CSS = """
<style>
/* Global background */
body {
    background-color: #F7F9FC;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #1A3C57 !important;
    color: white !important;
}

/* Sidebar text */
section[data-testid="stSidebar"] * {
    color: #E5E9F0 !important;
}

/* Main title */
h1 {
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    color: #1A1A1A;
}

/* Section headers */
h2 {
    font-family: 'Inter', sans-serif;
    color: #2A2A2A;
    margin-top: 1.5rem;
}

/* Chat message formatting */
.chat-box {
    background-color: #FFFFFF;
    border: 1px solid #D9DCE1;
    padding: 16px;
    border-radius: 8px;
    margin-bottom: 12px;
    font-size: 15px;
    line-height: 1.5;
}

.source-box {
    background-color: #F1F3F6;
    border-left: 4px solid #4E6C8A;
    padding: 10px;
    margin-top: 6px;
    font-size: 13px;
    border-radius: 4px;
}

/* Buttons */
.stButton button {
    background-color: #1A3C57 !important;
    color: white !important;
    border-radius: 6px;
    padding: 6px 14px;
    border: none;
}

.stButton button:hover {
    background-color: #254B6F !important;
}

/* Input text field */
.stTextInput input {
    border-radius: 6px;
    border: 1px solid #C8CDD4;
    padding: 10px;
}

/* Expander */
.streamlit-expanderHeader {
    font-size: 14px !important;
    font-weight: 500 !important;
    color: #1A1A1A !important;
}
</style>
"""


@st.cache_resource(show_spinner=True)
def build_knowledge_base():
    st.write("Building knowledge base...")
    articles = collect_articles()
    st.write(f"Fetched {len(articles)} articles.")

    all_chunks = []
    for art in articles:
        chunks = semantic_chunk(art)
        st.write(f"{art.source}: '{shorten(art.title, 60)}' — {len(chunks)} chunks")
        all_chunks.extend(chunks)

    index = ChunkIndex()
    index.build(all_chunks)
    return index, all_chunks


def answer_question(question: str, index: ChunkIndex):
    qa_prompt = load_prompt("qa_prompt")
    retrieved = index.query(question, k=5)

    context_pieces: List[str] = []
    for i, (chunk, score) in enumerate(retrieved):
        context_pieces.append(
            f"[CHUNK {i} | Score={score:.3f} | Article={chunk.article_id}]\n{chunk.text}\n"
        )

    context = "\n\n".join(context_pieces)

    user_prompt = (
        f"User question:\n{question}\n\n"
        f"Context:\n{context}\n\n"
        f"If context is insufficient, state that explicitly."
    )

    messages = format_system_user(qa_prompt, user_prompt)
    answer = chat_completion(messages)
    return answer, retrieved


def main():
    st.set_page_config(
        page_title="AI Regulation Insights Agent",
        layout="wide"
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    st.title("AI Regulation Insights Agent")
    st.write(
        f"""
        <p style='color:#444; font-size:16px; margin-top:-10px;'>
        Topic: <strong>{TOPIC}</strong>
        </p>
        """,
        unsafe_allow_html=True
    )

    with st.sidebar:
        st.header("Knowledge Base")
        if st.button("Rebuild Knowledge Base"):
            st.cache_resource.clear()
        index, chunks = build_knowledge_base()
        st.success("Knowledge base ready.")

    st.subheader("Ask a Question")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    question = st.text_input("Enter your question:")

    if st.button("Submit") and question.strip():
        with st.spinner("Processing..."):
            index, _ = build_knowledge_base()
            answer, retrieved = answer_question(question, index)

        st.session_state.chat_history.append(
            {
                "question": question,
                "answer": answer,
                "sources": [c.id for c, _ in retrieved],
            }
        )

    for turn in reversed(st.session_state.chat_history):
        st.markdown(
            f"<div class='chat-box'><strong>User:</strong> {turn['question']}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='chat-box'><strong>Assistant:</strong><br>{turn['answer']}</div>",
            unsafe_allow_html=True,
        )

        with st.expander("Show source chunk IDs"):
            st.write(turn["sources"])


if __name__ == "__main__":
    main()
