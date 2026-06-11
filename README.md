# 📊 Financial Report Q&A Engine

An end-to-end LangChain RAG system that answers natural language questions about real SEC filings (10-K / 10-Q) from Apple, Microsoft, and Google — with cited, grounded answers.

## 🎯 Live Demo

Ask questions like:
- *"What are Microsoft's main risk factors?"*
- *"What does Microsoft say about cloud growth?"*
- *"What is Microsoft's revenue from cloud services?"*

Every answer includes citations: `[MSFT | Filing: 10-Q | Page: 462]`

## 📊 Evaluation Results (RAGAS)

| Metric | Score |
|---|---|
| **Faithfulness** | **0.87** |
| **Answer Relevancy** | **0.74** |

## 🏗️ Architecture

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Framework | LangChain |
| Vector Database | ChromaDB |
| Embeddings | Cohere embed-english-v3.0 |
| Reranker | Cohere rerank-english-v3.0 |
| LLM | Groq LLaMA-3.3-70b-versatile |
| Evaluation | RAGAS |
| UI | Gradio |

## 🚀 Setup

```bash
git clone https://github.com/Harshita-madaan/financial-qa-engine
cd financial-qa-engine
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# Add COHERE_API_KEY and GROQ_API_KEY to .env
python -m src.downloader
python -m src.parser
python -m src.indexer
python gradio_app.py
```

## 🔑 Key Design Decisions

- **Two-stage retrieval** — ChromaDB semantic search (top 20) + Cohere reranker (top 5)
- **Citation-enforced prompting** — every claim must be cited with [Ticker | Filing | Page]
- **Graceful failure** — responds "Not found" instead of hallucinating
- **RAGAS evaluation** — quantified faithfulness and relevancy scores
