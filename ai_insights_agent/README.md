# AI Regulation Insights Agent

A Generative AI system that autonomously collects UK AI regulation content from trusted sources, organises it into a semantic knowledge layer, generates periodic summary reports, and supports retrieval-grounded conversational question answering.

This project is designed to demonstrate:
- **Semantic chunking** (meaning-based chunk boundaries)
- **Vector retrieval** (evidence selection before generation)
- **Multi-prompt orchestration** (reporting vs Q&A vs trend/change)
- **Context handling** (tight, controlled context windows)
- **Persistent memory** (reports saved across runs)

---

## Overview

The agent ingests trusted regulatory / policy content (primarily GOV.UK), cleans it, chunks it semantically, indexes chunks for retrieval, and then:
- Generates **periodic reports** (summary + takeaways + entities)
- Supports **conversational Q&A** grounded in retrieved chunks
- Supports **trend / change detection** using report history as memory

---

## Key Features

- Data collection from trusted sources (GOV.UK focus)
- Cleaned text + **semantic chunking** using sentence embeddings
- Vector search retrieval (top-k evidence chunks for each query)
- Periodic reports (100–150 word summary, 3–5 takeaways, entities)
- Multi-prompt design:
  - reporting prompt
  - Q&A prompt
  - trend/change prompt
- Persisted memory via saved report artifacts
- Streamlit UI (with visible retrieved evidence)
- CLI mode for quick interaction
- Unit tests + GitHub Actions CI

---

## Project Structure

```text
.
├── src/
│   ├── scraping/          # Data collection and parsing
│   ├── processing/        # Cleaning + semantic chunking
│   ├── retrieval/         # Vector indexing and search
│   ├── reporting/         # Report generation
│   ├── llm/               # LLM client + prompts
│   ├── data/              # Storage utilities
│   ├── app/               # CLI and Streamlit UI
│   └── config.py          # Configuration and paths
├── tests/                 # Focused unit tests
├── examples/              # Sample outputs (reports / conversations)
├── scripts/               # Utilities (e.g. reporting cycle runner)
├── requirements.txt
├── Dockerfile
└── README.md


---

## Setup

Environment:
Python 3.11 is recommended.

Create and activate a virtual environment:

Windows (PowerShell):
python -m venv .venv
.\.venv\Scripts\Activate.ps1

macOS / Linux:
python -m venv .venv
source .venv/bin/activate

Install dependencies:
pip install -r requirements.txt

Download required NLTK data (first run only):
python -m nltk.downloader punkt

---

## Configuration

Set your OpenAI API key as an environment variable.

Windows:
setx OPENAI_API_KEY "your-key-here"

macOS / Linux:
export OPENAI_API_KEY="your-key-here"

---

## Running the System

Run a reporting cycle:
python -m src.reporting.run_report

This performs ingestion, semantic chunking, indexing, report generation, and persists the report for later reuse.

Run the Streamlit UI:
streamlit run src/app/ui_app.py

Run the CLI:
python -m src.app.cli

Example questions:
- What’s new in UK AI regulation?
- What’s happening nowadays?
- How has the picture changed since last week?

---

## Multi-Prompt Strategy

The system uses separate prompts for:
- Reporting
- Conversational question answering
- Trend and change detection

Each prompt has a single responsibility to keep reasoning predictable and explainable.

---

## Context Handling and Memory

Short-term memory:
Recent chat turns for conversational coherence.

Long-term memory:
Persisted report artifacts generated during periodic runs.

Reports act as curated memory checkpoints and are reused for trend and change analysis.

---

## Semantic Chunking

Text is split based on meaning rather than fixed size:
- Sentence-level tokenisation
- Sentence embeddings
- Semantic grouping with similarity thresholds
- Controlled overlap for continuity

This improves retrieval relevance and answer grounding.

---

## Testing

Run tests:
pytest

Tests cover semantic chunking behaviour, retrieval relevance, and report persistence.
They also run automatically via GitHub Actions CI.

---

## Docker

Build the image:
docker build -t ai-insights-agent .

Run the container:
docker run -p 8501:8501 ai-insights-agent

---

## Example Outputs

The examples directory contains:
- Sample generated reports
- Example conversation transcripts
- Trend and change detection outputs

---

## Notes

- Lightweight but production-aligned
- Modular and extensible
- Clear separation between ingestion, retrieval, generation, and memory
