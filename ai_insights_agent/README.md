AI Regulation Insights Agent

A Generative AI system that autonomously collects UK AI regulation content from trusted sources, organises it into a semantic knowledge layer, generates periodic summary reports, and supports retrieval-grounded conversational question answering.

The system is designed to demonstrate semantic chunking, vector retrieval, multi-prompt orchestration, context handling, and persistent memory over time.

Overview

This project implements an end-to-end Insights Agent focused on UK AI regulation and governance. It continuously ingests official policy and guidance content, structures it into semantically meaningful units, and enables both periodic reporting and conversational querying over the collected knowledge.

Key capabilities include:

Automated data collection from trusted sources (primarily GOV.UK)

Semantic chunking based on sentence embeddings

Vector-based retrieval for grounded question answering

Periodic report generation with summaries, takeaways, and entities

Multi-prompt LLM strategy for reporting, Q&A, and trend detection

Persistent memory via saved reports

Interactive chat interface (CLI and Streamlit UI)

Project Structure
.
├── src/
│   ├── scraping/          # Data collection and parsing
│   ├── processing/        # Cleaning and semantic chunking
│   ├── retrieval/         # Vector indexing and search
│   ├── reporting/         # Report generation and persistence
│   ├── llm/               # Prompt templates and orchestration
│   ├── data/              # Storage utilities
│   ├── app/               # CLI and Streamlit UI
│   └── config.py          # Configuration and paths
├── tests/                 # Focused unit tests
├── examples/              # Sample reports and outputs
├── requirements.txt
├── Dockerfile
└── README.md


The codebase is structured so that ingestion, retrieval, generation, and memory are cleanly separated and easy to reason about.

Setup
1. Environment

Python 3.11 is recommended.

Create and activate a virtual environment:

python -m venv .venv
source .venv/bin/activate   # macOS / Linux
.venv\Scripts\activate      # Windows

2. Install dependencies
pip install -r requirements.txt


If you are running locally for the first time, download required NLTK data:

python -m nltk.downloader punkt

3. Configuration

Set your OpenAI API key as an environment variable:

export OPENAI_API_KEY="your-key-here"   # macOS / Linux
setx OPENAI_API_KEY "your-key-here"     # Windows


No other configuration is required for local execution.

Running the System
1. Data ingestion and report generation

Run a reporting cycle to fetch articles, chunk them semantically, and generate a report:

python -m src.reporting.run_report


This will:

Fetch recent regulatory content

Perform semantic chunking

Generate a structured report (summary, takeaways, entities)

Persist the report for later use

Generated reports are saved and reused as long-term memory.

2. Conversational interface (CLI)

To interact with the system via the command line:

python -m src.app.cli


You can ask both specific and open-ended questions such as:

“What’s new in UK AI regulation?”

“What’s happening nowadays?”

“How has the picture changed since last week?”

Answers are grounded in retrieved chunks and recent reports.

3. Web interface (Streamlit)

To launch the Streamlit UI:

streamlit run src/app/ui_app.py


The UI provides:

A chat interface for conversational Q&A

Visibility into retrieved chunks used for grounding

Access to generated reports and report history

This makes system behaviour observable and inspectable.

Multi-Prompt Strategy

The system uses multiple prompts, each with a single responsibility:

Reporting prompt
Generates periodic summaries, key takeaways, and entities from newly ingested content.

Conversational Q&A prompt
Answers user questions using only retrieved chunks and limited conversational context.

Trend / change-detection prompt
Compares the latest report with previous reports to describe how the situation has evolved.

Prompt separation ensures predictable behaviour and avoids cross-task interference.

Context Handling and Memory

Short-term memory
Limited to recent conversational turns to maintain coherence.

Long-term memory
Consists of persisted report artefacts generated during periodic runs.

Reports act as curated memory checkpoints and are explicitly reused for trend analysis and “what changed?” questions.

The system avoids storing full chat transcripts as long-term memory to prevent noise accumulation.

Semantic Chunking

Text is chunked based on semantic meaning rather than fixed size:

Cleaned text is split into sentences

Each sentence is embedded

Adjacent sentences are grouped when semantic similarity exceeds a threshold

Controlled overlap is added to preserve continuity

This produces chunks aligned with conceptual boundaries in regulatory text and improves retrieval quality.

Testing

Run the test suite with:

pytest


The tests cover:

Semantic chunking behaviour

Retrieval relevance

Report persistence and loading

A GitHub Actions CI workflow runs these tests automatically on each push.

Docker

A Dockerfile is included to ensure consistent environments.

To build the image:

docker build -t ai-insights-agent .


To run the Streamlit UI:

docker run -p 8501:8501 ai-insights-agent

Example Outputs

The examples/ directory contains:

Sample generated reports

Example conversation transcripts

Demonstrations of trend and change-detection behaviour

These artefacts illustrate how memory and retrieval are reused across runs.

Notes

The system is intentionally lightweight but production-aligned.

All components are modular and replaceable.

Retrieval, generation, and memory are explicitly separated for clarity and robustness.