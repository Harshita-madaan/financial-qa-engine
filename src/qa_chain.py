"""
src/qa_chain.py — RAG chain with ChromaDB + Groq llama-3.3-70b.
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


def _build_context(docs):
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
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K_FINAL})
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=LLM_TEMPERATURE,
        api_key=os.getenv("GROQ_API_KEY"),
    )
    return {"retriever": retriever, "llm": llm}


def ask(chain, question):
    docs = chain["retriever"].invoke(question)
    log.info(f"Retrieved {len(docs)} chunks")

    context = _build_context(docs)
    prompt = PROMPT.format(context=context, question=question)

    response = chain["llm"].invoke(prompt)
    answer = response.content if hasattr(response, "content") else str(response)

    return {
        "answer": answer,
        "sources": [doc.metadata for doc in docs],
        "docs": docs,
    }