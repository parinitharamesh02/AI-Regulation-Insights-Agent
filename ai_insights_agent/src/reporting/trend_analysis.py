from typing import Optional

from ..config import TOPIC, EXAMPLES_DIR
from ..logging_utils import get_logger
from ..llm import chat_completion, load_prompt, format_system_user
from ..data.storage import load_all_reports

logger = get_logger(__name__)


def build_trend_analysis() -> Optional[str]:
    """
    Compare the latest report with the previous one using the trend prompt.
    """
    reports = load_all_reports()
    if len(reports) < 2:
        return None

    latest = reports[-1]
    previous = reports[-2]

    trend_prompt = load_prompt("trend")

    user_prompt = (
        f"You are analysing changes in the topic: {TOPIC}.\n\n"
        f"Latest report ({latest.created_at.isoformat()}):\n"
        f"SUMMARY:\n{latest.summary}\n\n"
        f"TAKEAWAYS:\n{latest.takeaways}\n\n"
        f"Previous report ({previous.created_at.isoformat()}):\n"
        f"SUMMARY:\n{previous.summary}\n\n"
        f"TAKEAWAYS:\n{previous.takeaways}\n\n"
        "Describe what changed between these, as per the instructions."
    )

    messages = format_system_user(trend_prompt, user_prompt)
    answer = chat_completion(messages)
    return answer


def save_trend_analysis_to_examples() -> Optional[str]:
    analysis = build_trend_analysis()
    if analysis is None:
        logger.info("Not enough historical reports for trend analysis.")
        return None

    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EXAMPLES_DIR / "trend_analysis.txt"
    out_path.write_text(analysis, encoding="utf-8")
    logger.info("Trend analysis saved to %s", out_path)
    return str(out_path)
