import json
from datetime import datetime
from typing import List, Optional


from .config import REPORTS_DIR, TOPIC, EXAMPLES_DIR
from .llm_client import load_prompt, format_system_user, chat_completion
from .models import Report


def _load_reports() -> List[Report]:
    if not REPORTS_DIR.exists():
        return []

    reports: List[Report] = []
    for path in sorted(REPORTS_DIR.glob("report_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        created_at = datetime.fromisoformat(str(data["created_at"]))
        reports.append(
            Report(
                id=data["id"],
                created_at=created_at,
                topic=data["topic"],
                summary=data["summary"],
                takeaways=data["takeaways"],
                entities=data["entities"],
            )
        )
    # sort by created_at just in case
    reports.sort(key=lambda r: r.created_at)
    return reports


def build_trend_analysis() -> Optional[str]:
    """
    Compare the latest report with the previous one (if available)
    using the trend_prompt. Returns None if not enough history.
    """
    reports = _load_reports()
    if len(reports) < 2:
        return None  # Not enough history

    latest = reports[-1]
    previous = reports[-2]

    trend_prompt = load_prompt("trend_prompt")

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

    messages = format_system_user(
        system_prompt=trend_prompt,
        user_prompt=user_prompt,
    )
    answer = chat_completion(messages)
    return answer


def main():
    analysis = build_trend_analysis()
    if analysis is None:
        print("[!] Not enough historical reports to do trend analysis.")
        print("    Generate at least two reports by running `python -m src.reporting` multiple times.")
        return

    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EXAMPLES_DIR / "trend_analysis.txt"
    out_path.write_text(analysis, encoding="utf-8")

    print("[*] Trend analysis generated:")
    print()
    print(analysis)
    print()
    print(f"[*] Saved to {out_path}")


if __name__ == "__main__":
    main()
