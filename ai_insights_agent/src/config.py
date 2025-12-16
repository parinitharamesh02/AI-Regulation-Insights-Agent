from pathlib import Path
import os

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = DATA_DIR / "reports"
EXAMPLES_DIR = BASE_DIR / "examples"
CHAT_DIR = DATA_DIR / "chat"

for p in (DATA_DIR, RAW_DIR, PROCESSED_DIR, REPORTS_DIR, EXAMPLES_DIR, CHAT_DIR):
    p.mkdir(parents=True, exist_ok=True)

# Topic
TOPIC = "AI regulation and governance in the UK"

# Sources
BBC_SOURCE_NAME = "BBC"
GOVUK_SOURCE_NAME = "GOV.UK"

BBC_ARTICLE_URLS = [
    # Optional BBC AI/tech policy URLs
]

GOVUK_ARTICLE_URLS = [
    "https://www.gov.uk/government/publications/ai-regulation-a-pro-innovation-approach",
    "https://www.gov.uk/government/publications/online-safety-act-explainer/online-safety-act-explainer",
    "https://www.gov.uk/government/publications/ai-opportunities-action-plan/ai-opportunities-action-plan",
    "https://www.gov.uk/government/consultations/"
    "legal-framework-for-using-facial-recognition-in-law-enforcement/"
    "options-assessment-consultation-on-a-new-legal-framework-for-law-enforcement-"
    "use-of-biometrics-facial-recognition-and-similar-technologies",
]

# Embeddings
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# LLM model names (can be overridden by env vars)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
