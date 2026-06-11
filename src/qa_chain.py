"""
src/qa_chain.py — RAG chain with ChromaDB + Cohere Reranker + Groq LLaMA-3.3-70b.
Two-stage retrieval:
  1. ChromaDB semantic search → top 20 candidates
  2. Cohere reranker → top 5 most relevant
  3. Groq LLaMA generates cited answer
"""
import logging
import os
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from config import TOP_K_FAISS, TOP_K_FINAL, LLM_TEMPERATURE

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a precise financial analyst assistant.
Answer the question using ONLY the information in the context below.

Rules:
- Every number or claim MUST be followed by its source: [Ticker | Filing | Page]
- If the answer is NOT in the context, respond: "Not found in the provided filings."
- Do not speculate or use outside knowledge.
- Keep the answer concise and structured.

Context:
{context}

Question: {question}

Answer:""",
)


def _cohere_rerank(query: str, docs: list, top_n: int = 5) -> list:
    """
    Rerank documents using Cohere cross-encoder.
    Returns top_n most relevant docs in order.
    """
    try:
        import cohere
        co = cohere.Client(os.getenv("COHERE_API_KEY"))
        texts = [doc.page_content for doc in docs]

        response = co.rerank(
            model="rerank-english-v3.0",
            query=query,
            documents=texts,
            top_n=top_n,
        )

        reranked = [docs[r.index] for r in response.results]
        log.info(f"Reranker: {len(docs)} → {len(reranked)} docs")
        return reranked

    except Exception as e:
        log.warning(f"Reranker failed ({e}), using original order.")
        return docs[:top_n]


def _build_context(docs: list) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        m = doc.metadata
        header = (f"[Source {i}] "
                  f"Ticker: {m.get('ticker','?')} | "
                  f"Filing: {m.get('filing_type','?')} | "
                  f"Page: {m.get('page','?')}")
        parts.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def build_chain():
    from src.indexer import load_index
    vectorstore = load_index()

    # Stage 1: broad retrieval — get top 20 candidates
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K_FAISS})

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=LLM_TEMPERATURE,
        api_key=os.getenv("GROQ_API_KEY"),
    )
    return {"retriever": retriever, "llm": llm}


def ask(chain, question: str) -> dict:
    # Stage 1 — semantic search (top 20)
    docs = chain["retriever"].invoke(question)
    log.info(f"Stage 1 — ChromaDB retrieved {len(docs)} candidates")

    # Stage 2 — rerank (top 5)
    docs = _cohere_rerank(question, docs, top_n=TOP_K_FINAL)
    log.info(f"Stage 2 — Reranker selected {len(docs)} docs")

    # Stage 3 — generate answer
    context = _build_context(docs)
    prompt = PROMPT.format(context=context, question=question)
    response = chain["llm"].invoke(prompt)
    answer = response.content if hasattr(response, "content") else str(response)

    return {
        "answer": answer,
        "sources": [doc.metadata for doc in docs],
        "docs": docs,
    }