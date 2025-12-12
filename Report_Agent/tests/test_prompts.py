from pathlib import Path

from src.config import BASE_DIR


def test_prompt_files_exist() -> None:
    prompts_dir = BASE_DIR / "src" / "prompts"
    assert (prompts_dir / "qa_prompt.txt").exists()
    assert (prompts_dir / "report_prompt.txt").exists()
    assert (prompts_dir / "trend_prompt.txt").exists()


def test_prompts_are_not_empty() -> None:
    prompts_dir = BASE_DIR / "src" / "prompts"

    for name in ["qa_prompt.txt", "report_prompt.txt", "trend_prompt.txt"]:
        text = (prompts_dir / name).read_text(encoding="utf-8")
        assert text.strip(), f"{name} should not be empty"
