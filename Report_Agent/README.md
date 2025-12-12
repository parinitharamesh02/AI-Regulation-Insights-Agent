# AI Regulation Insights Agent
A production-oriented Generative AI system that autonomously ingests regulatory information, constructs a semantic knowledge base, and provides context-grounded question answering, reporting, and trend analysis.  
This project was developed as part of a technical assessment and reflects the engineering principles typically used in enterprise LLMOps and MLOps environments.

---

## 1. Overview

This system monitors UK AI regulation by:

- Collecting articles from trusted sources (GOV.UK, BBC Technology where permitted)
- Cleaning and structuring the text
- Producing semantically meaningful chunks
- Indexing content using vector embeddings
- Enabling retrieval-augmented question answering
- Generating periodic summary reports with key entities
- Performing trend analysis over historical reports
- Exposing both a CLI interface and a Streamlit UI

The solution is modular, containerised, and aligned with scalable deployment patterns used on AWS (ECS Fargate, Lambda-based scheduled tasks, or SageMaker endpoints where appropriate).

---

## 2. Architecture Summary

The system consists of the following components:

### Data Ingestion  
- Fetches raw HTML from GOV.UK and BBC articles  
- Cleans boilerplate, navigation text, and irrelevant content  
- Normalises and stores raw text under `data/raw/`

### Semantic Chunking  
- Splits text using sentence segmentation combined with semantic grouping  
- Preserves conceptual boundaries in policy documents  
- Outputs structured chunks under `data/processed/`

### Vector Retrieval  
- Embeds chunks using a sentence-transformer model  
- Builds an in-memory vector index for top-k similarity search  
- Returns relevant context for question answering

### LLM Orchestration  
- Separate prompts for:
  - Report summarisation
  - Conversational Q&A
  - Entity extraction and trend analysis
- Strict anti-hallucination constraints to ensure responses remain grounded in retrieved data

### Reporting and Memory  
- Generates timestamped reports (JSON) with summaries, key takeaways, and entities  
- Stores report history under `data/reports/`  
- Computes differences between reports for trend analysis

### User Interfaces  
- CLI assistant (`python -m src.chat_cli`)  
- Streamlit UI (`streamlit run src/ui_app.py`)

---

## 3. Project Structure

Report_Agent/
│
├── src/
│   ├── scraper.py
│   ├── chunking.py
│   ├── index.py
│   ├── logging_utils.py
│   ├── llm_client.py
│   ├── models.py
│   ├── reporting.py
│   ├── trend.py
│   ├── chat_cli.py
│   ├── ui_app.py
│   └── prompts/
│       ├── qa_prompt.txt
│       ├── report_prompt.txt
│       └── entity_prompt.txt
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── reports/
│
├── examples/
│   ├── sample_report.json
│   └── trend_analysis.txt
│
├── Dockerfile
├── requirements.txt
└── .github/workflows/ci.yml


## 4. Installation and Setup

### Install Dependencies
pip install -r requirements.txt

bash
Copy code

### Configure Environment Variables
Create a `.env` file in the project root:
OPENAI_API_KEY=your_key_here

graphql
Copy code

### Run the CLI Assistant
python -m src.chat_cli

graphql
Copy code

### Run the Web UI
streamlit run src/ui_app.py

shell
Copy code

### Generate a Report
python -m src.reporting

shell
Copy code

### Run Trend Analysis
python -m src.trend

yaml
Copy code

---

## 5. Docker Support

A production-ready Dockerfile is included:

docker build -t ai-report-agent .
docker run -it --rm ai-report-agent

yaml
Copy code

The image uses a slim Python base, installs only required dependencies, and runs the CLI by default.  
This image structure is suitable for ECS Fargate deployment or Lambda packaging.

---

## 6. CI Pipeline

A GitHub Actions workflow is provided under `.github/workflows/ci.yml` and performs:

- Dependency installation  
- Syntax and import checks  
- Docker build  
- Optional extension for ECR deployment or security scanning  

This reflects standard CI patterns for containerised GenAI applications.

---

## 7. AWS Deployment Considerations

Although the take-home does not require cloud deployment, the application is structured so it can be deployed on AWS with minimal changes.

Recommended architecture:

- ECS Fargate service for the Streamlit UI or FastAPI API extension  
- EventBridge scheduled rule triggering:
  - ECS Task or Lambda for periodic reporting  
- S3 for report storage and long-term archival  
- CloudWatch for metrics and centralised logging  
- IAM roles with least-privilege access to API keys, S3 buckets, and logs  
- (Optional) SageMaker endpoints for embedding or LLM inference  
- (Optional) OpenSearch for scalable vector search

This reflects the patterns used in production MLOps/LLMOps environments.

---

## 8. Extensibility and Future Enhancements

The system can be extended to support:

- Multi-agent orchestration (retrieval agent, reporting agent, governance agent)
- Additional regulatory sources (ICO, Ofcom, DSIT, Parliamentary committees)
- Advanced memory (long-term entity tracking across months)
- Prompt injection defences and output validation
- Model monitoring (latency, drift, hallucination rate)
- Automated lineage and metadata tracking using MLflow or SageMaker Model Registry
- Distributed ingestion pipelines using AWS Step Functions or Airflow

---

## 9. Example Outputs

Sample generated report and trend analysis files are included in the `examples/` directory.

---

## 10. Notes

This codebase has been developed with an emphasis on clarity, modularity, and production alignment.  
The intention is to reflect how a regulatory intelligence or governance-oriented GenAI system would be implemented in a real client engagement, following MLOps and LLMOps best practices.

