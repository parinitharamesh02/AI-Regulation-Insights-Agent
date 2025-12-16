from datetime import datetime
from typing import List, Tuple

from ..config import TOPIC
from ..logging_utils import get_logger
from ..models import Chunk, ConversationTurn
from ..scraping import collect_articles
from ..processing.chunking import semantic_chunk
from ..retrieval.index import ChunkIndex
from ..llm import chat_completion, load_prompt, format_system_user

logger = get_logger(__name__)


def build_knowledge_base() -> Tuple[ChunkIndex, List[Chunk]]:
    """
    Ingest articles, perform semantic chunking, and build a vector index.
    """
    logger.info("Collecting articles for CLI assistant...")
    articles = collect_articles()
    logger.info("Fetched %d articles", len(articles))

    all_chunks: List[Chunk] = []
    for art in articles:
        chunks = semantic_chunk(art)
        logger.info(
            "Source=%s, title='%s', chunks=%d",
            art.source,
            art.title[:60],
            len(chunks),
        )
        all_chunks.extend(chunks)

    index = ChunkIndex()
    index.build(all_chunks)
    logger.info("Vector index built.")

    return index, all_chunks


def build_qa_prompt_with_history(
    question: str,
    retrieved: List[Tuple[Chunk, float]],
    history: List[ConversationTurn],
) -> str:
    # Last N turns
    recent = history[-3:]

    history_parts: List[str] = []
    for turn in recent:
        history_parts.append(f"User: {turn.user_question}")
        history_parts.append(f"Assistant: {turn.answer}")
    history_str = "\n".join(history_parts) if history_parts else "(no prior conversation)"

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
        "Use the conversation history only for additional context and clarification. "
        "Base factual claims strictly on the retrieved chunks. "
        "If the context is insufficient, state that explicitly."
    )
    return user_prompt


def answer_question(
    question: str,
    index: ChunkIndex,
    history: List[ConversationTurn],
) -> str:
    qa_prompt = load_prompt("qa")
    retrieved = index.query(question, k=5)

    user_prompt = build_qa_prompt_with_history(question, retrieved, history)
    messages = format_system_user(qa_prompt, user_prompt)

    logger.info("Answering question via CLI: %s", question)
    answer = chat_completion(messages)

    used_chunk_ids = [c.id for c, _ in retrieved]
    history.append(
        ConversationTurn(
            timestamp=datetime.utcnow(),
            user_question=question,
            answer=answer,
            used_chunk_ids=used_chunk_ids,
            used_report_ids=[],
        )
    )
    return answer


def main() -> None:
    logger.info("Starting AI Regulation Insights Agent (CLI)...")
    index, _ = build_knowledge_base()
    history: List[ConversationTurn] = []

    print()
    print("=== AI Regulation Insights Agent (CLI) ===")
    print(f"Topic: {TOPIC}")
    print("Type your questions (or 'exit' to quit).")
    print()

    while True:
        q = input("You: ").strip()
        if q.lower() in {"exit", "quit"}:
            print("Exiting.")
            break
        if not q:
            continue

        print("[*] Thinking...")
        try:
            ans = answer_question(q, index, history)
            print("\nAssistant:\n")
            print(ans)
            print()
        except Exception as e:
            logger.error("Error while answering question: %s", e)
            print("Sorry, something went wrong while answering your question.")


if __name__ == "__main__":
    main()
