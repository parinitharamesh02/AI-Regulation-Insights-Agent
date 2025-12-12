AI Regulation Insights Agent

A lightweight agent that ingests trusted UK policy sources (GOV.UK), builds a semantic knowledge base, generates periodic reports, and provides a grounded conversational interface.

This project implements the core requirements of the Data Reply – AI Engineer Take-Home Test using clean modular components and a professional UI.

Features
Data ingestion & processing

Scrapes curated GOV.UK AI-related policy pages.

Cleans and normalises content into structured Article objects.

Performs semantic chunking using sentence embeddings (not fixed-size).

Stores articles, chunks, and reports under data/.

Retrieval-Augmented Generation (RAG)

Embeds chunks using all-MiniLM-L6-v2.

FAISS vector index for fast similarity search.

Supports grounded Q&A with retrieved sources shown in UI.

Reporting pipeline

100–150 word structured summaries.

Key takeaways + entity extraction (organisations, people, locations, terms).

Periodic reports stored as JSON.

Trend analysis over adjacent reports ("what changed?").

Conversational UI

Clean Streamlit dashboard with three tabs:

Chat: grounded Q&A + trend routing.

Reports: generate & inspect historical reports.

Knowledge Base: inspect article/chunk counts and examples.

Persisted chat history across sessions.

Project Structure
src/
  scraping/       # GOV.UK parsing & collection
  processing/     # embeddings + semantic chunking
  retrieval/      # FAISS index
  llm/            # OpenAI client + prompts
  reporting/      # report + trend generation
  app/            # Streamlit UI and CLI


Runtime data lives in data/:

raw/ articles

processed/ chunks

reports/ structured reports

chat/ persisted chat histories

Setup
1. Install
python -m venv .venv
source .venv/bin/activate   # or .\.venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt

2. Environment

Create .env:

OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4o-mini

Usage
Run a reporting cycle

Fetch, chunk, embed, and generate a report:

python scripts/run_reporting_cycle.py

Run trend analysis

Requires ≥2 reports:

python scripts/run_trend_analysis.py

Run the UI
streamlit run src/app/ui_app.py

Optional CLI
python -m src.app.cli

Multi-Prompt Workflow

The system uses three specialised prompts:

qa.txt — conversational Q&A grounded in retrieved chunks

report.txt — batch summarisation & entity extraction

trend.txt — “what changed” reasoning over past reports

Trend-like questions (“what changed?”, “how is it different now?”) are automatically routed to the trend prompt.

Semantic Chunking

Chunking is based on:

sentence tokenisation

sentence-level embeddings

centroid similarity threshold for grouping

slight sentence overlap for continuity

This ensures chunks follow semantic boundaries, not arbitrary fixed sizes — ideal for policy and regulation text.

AWS/MLOps Ready

The repo maps cleanly to a production pipeline:

Scraping + reporting → Lambda / Fargate / Step Functions

Chunking + indexing → SageMaker Processing or EKS job

Artifacts → S3

Chat API → ECS Fargate / Lambda + API Gateway

Logs & metrics → CloudWatch

Summary

This agent demonstrates:

End-to-end RAG pipeline

Semantic chunking

Multi-prompt orchestration

Trend analysis as temporal memory

Grounded conversational interface

Modular, production-oriented architecture

Designed to be clear, inspectable, and easy to extend.
