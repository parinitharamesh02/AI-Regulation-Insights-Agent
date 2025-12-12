from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EXAMPLES_DIR = BASE_DIR / "examples"
REPORTS_DIR = DATA_DIR / "reports"


# Topic & sources
TOPIC = "AI regulation in the UK"

BBC_SOURCE_NAME = "BBC"
GOVUK_SOURCE_NAME = "GOV.UK"

BBC_ARTICLE_URLS = [
    # AI, biometrics and policing – great for talking about real-world, high-risk AI use cases
    #"https://www.bbc.co.uk/news/articles/c62lq580696o",

    # The example URL given in the task is also genuinely about AI and regulation,
    # so it's safe to keep as an additional tech/regulation piece
    "https://www.bbc.co.uk/news/technology-67297110",
]

GOVUK_ARTICLE_URLS = [
    # Core UK AI regulation white paper – absolutely central for any AI governance agent
    "https://www.gov.uk/government/publications/ai-regulation-a-pro-innovation-approach",

    # Online Safety Act explainer – ties to content moderation, platforms and safety-by-design
    "https://www.gov.uk/government/publications/online-safety-act-explainer/online-safety-act-explainer",

    # AI Opportunities Action Plan – focuses on how government wants to *use* AI across sectors
    "https://www.gov.uk/government/publications/ai-opportunities-action-plan/ai-opportunities-action-plan",

    # Consultation on biometrics & facial recognition – nice bridge between AI, policing and data protection
    "https://www.gov.uk/government/consultations/"
    "legal-framework-for-using-facial-recognition-in-law-enforcement/"
    "options-assessment-consultation-on-a-new-legal-framework-for-law-enforcement-"
    "use-of-biometrics-facial-recognition-and-similar-technologies",
]

DATA_REPLY_REFERENCE_URLS = [
    "https://www.reply.com/data-reply/en/generative-ai-solutions",
    "https://aws.amazon.com/blogs/machine-learning/"
    "responsible-ai-in-action-how-data-reply-red-teaming-supports-generative-ai-safety-on-aws/",
]


# Embedding model
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# LLM model name (for docs only; exact value depends on your provider)
LLM_MODEL_NAME = "gpt-4o-mini"  # or any capable chat model
