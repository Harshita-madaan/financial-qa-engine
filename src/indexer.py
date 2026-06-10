"""
src/indexer.py — chunk, embed with Cohere, store in ChromaDB.
Run: python -m src.indexer
"""
import json
import logging
import os
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain_chroma import Chroma
from config import DATA_PROCESSED, INDEX_DIR, CHUNK_SIZE, CHUNK_OVERLAP

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

CHROMA_DIR = str(INDEX_DIR / "chroma")


def get_embeddings():
    return CohereEmbeddings(
        model="embed-english-v3.0",
        cohere_api_key=os.getenv("COHERE_API_KEY"),
    )


def load_parsed_pages():
    path = DATA_PROCESSED / "parsed_pages.json"
    if not path.exists():
        raise FileNotFoundError("No parsed pages found. Run parser.py first.")
    with open(path) as f:
        return json.load(f)


def chunk_pages(pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    texts, metas = [], []
    for page in pages:
        raw = page.get("text", "").strip()
        if not raw:
            continue
        for chunk in splitter.split_text(raw):
            if len(chunk) < 80:
                continue
            texts.append(chunk)
            metas.append({
                "ticker":      page.get("ticker", ""),
                "filing_type": page.get("filing_type", ""),
                "accession":   page.get("accession", ""),
                "page":        str(page.get("page", 0)),
                "source":      page.get("source", ""),
                "text":        chunk,
            })
    log.info(f"Pages: {len(pages)}  →  Chunks: {len(texts)}")
    return texts, metas


def build_index(texts, metas):
    import shutil
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # Clear old index if exists
    if Path(CHROMA_DIR).exists():
        shutil.rmtree(CHROMA_DIR)
        log.info("Cleared old index.")

    embeddings = get_embeddings()
    log.info(f"Embedding {len(texts)} chunks with Cohere embed-english-v3.0 ...")

    BATCH = 90  # Cohere free tier limit is 100/min
    import time
    vectorstore = None

    for i in range(0, len(texts), BATCH):
        batch_texts = texts[i:i+BATCH]
        batch_metas = metas[i:i+BATCH]
        log.info(f"  Batch {i//BATCH + 1}/{(len(texts)-1)//BATCH + 1}: chunks {i} to {i+len(batch_texts)}")

        if vectorstore is None:
            vectorstore = Chroma.from_texts(
                texts=batch_texts,
                embedding=embeddings,
                metadatas=batch_metas,
                persist_directory=CHROMA_DIR,
            )
        else:
            vectorstore.add_texts(texts=batch_texts, metadatas=batch_metas)

        if i + BATCH < len(texts):
            time.sleep(12)  # stay within free tier rate limit

    with open(INDEX_DIR / "metadata.json", "w") as f:
        json.dump(metas, f, indent=2)

    log.info(f"ChromaDB index saved → {CHROMA_DIR}")
    return vectorstore


def load_index():
    embeddings = get_embeddings()
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )
    log.info("ChromaDB index loaded.")
    return vectorstore


if __name__ == "__main__":
    pages = load_parsed_pages()
    texts, metas = chunk_pages(pages)
    build_index(texts, metas)