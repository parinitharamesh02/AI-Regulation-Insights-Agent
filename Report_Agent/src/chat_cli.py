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


def answer_question(question: str, index: ChunkIndex) -> str:
    """
    Retrieve relevant chunks and answer a user question via qa_prompt.
    """
    qa_prompt = load_prompt("qa_prompt")
    retrieved = index.query(question, k=5)

    context_parts = []
    for i, (chunk, score) in enumerate(retrieved):
        context_parts.append(
            f"[CHUNK {i} | score={score:.3f} | article_id={chunk.article_id}]\n{chunk.text}\n"
        )
    context = "\n\n".join(context_parts)

    user_prompt = (
        f"User question:\n{question}\n\n"
        f"Context chunks:\n{context}\n\n"
        f"If the context is insufficient, state that explicitly."
    )

    messages = format_system_user(
        system_prompt=qa_prompt,
        user_prompt=user_prompt,
    )

    logger.info("Answering question: %s", question)
    answer = chat_completion(messages)
    return answer


def main():
    logger.info("Starting AI Regulation News Assistant (CLI)...")
    index, chunks = build_knowledge_base()

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
            ans = answer_question(q, index)
            print("\nAssistant:\n")
            print(ans)
            print()
        except Exception as e:
            logger.error("Error while answering question: %s", e)
            print("Sorry, something went wrong while answering your question.")


if __name__ == "__main__":
    main()
