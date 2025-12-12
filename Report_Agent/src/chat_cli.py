from typing import List, Tuple

from .scraper import collect_articles
from .chunking import semantic_chunk
from .index import ChunkIndex
from .llm_client import chat_completion, load_prompt, format_system_user
from .config import TOPIC
from .models import Chunk
from .logging_utils import get_logger

logger = get_logger(__name__)


def build_knowledge_base() -> Tuple[ChunkIndex, List[Chunk]]:
    """
    Ingest articles, perform semantic chunking, and build a vector index.

    Returns:
        (ChunkIndex, List[Chunk]): the vector index and the flat list of chunks.
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

    logger.info("Total chunks: %d", len(all_chunks))

    index = ChunkIndex()
    index.build(all_chunks)
    logger.info("Vector index built.")

    return index, all_chunks


def build_qa_prompt_with_history(
    question: str,
    retrieved: List[Tuple[Chunk, float]],
    chat_history: List[dict],
) -> str:
    """
    Build the user prompt including last N turns of conversation and retrieved chunks.
    """
    # Limit conversation history to last 3 turns
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
        "Use the conversation history only for additional context and clarification. "
        "Base factual claims strictly on the retrieved chunks. "
        "If the context is insufficient, state that explicitly."
    )
    return user_prompt


def answer_question(
    question: str,
    index: ChunkIndex,
    chat_history: List[dict],
) -> str:
    """
    Retrieve relevant chunks and answer a user question via qa_prompt, using
    both conversational history and retrieved context.
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

    logger.info("Answering question: %s", question)
    answer = chat_completion(messages)
    return answer


def main() -> None:
    logger.info("Starting AI Regulation News Assistant (CLI)...")
    index, _ = build_knowledge_base()

    chat_history: List[dict] = []

    print()
    print("=== AI Regulation News Assistant ===")
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
            ans = answer_question(q, index, chat_history)
            print("\nAssistant:\n")
            print(ans)
            print()

            chat_history.append(
                {
                    "question": q,
                    "answer": ans,
                }
            )
        except Exception as e:
            logger.error("Error while answering question: %s", e)
            print("Sorry, something went wrong while answering your question.")


if __name__ == "__main__":
    main()
