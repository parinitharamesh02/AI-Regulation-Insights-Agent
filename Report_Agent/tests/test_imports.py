def test_import_core_modules() -> None:
    """
    Basic smoke test to ensure core modules import successfully.
    This avoids hitting external services (no scraping or OpenAI calls).
    """
    import src.scraper  # noqa: F401
    import src.chunking  # noqa: F401
    import src.index  # noqa: F401
    import src.reporting  # noqa: F401
    import src.trend  # noqa: F401
    import src.chat_cli  # noqa: F401
    import src.ui_app  # noqa: F401
    import src.llm_client  # noqa: F401
    import src.models  # noqa: F401
