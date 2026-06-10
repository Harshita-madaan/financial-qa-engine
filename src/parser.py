"""
src/parser.py — extract clean text from SEC filings.
Run: python -m src.parser
"""
import json
import logging
import re
from html.parser import HTMLParser
from pathlib import Path
import pdfplumber
from config import DATA_RAW, DATA_PROCESSED

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def _clean(text):
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _meta_from_path(path):
    parts = path.parts
    try:
        idx = parts.index("sec-edgar-filings")
        return {
            "ticker":      parts[idx + 1],
            "filing_type": parts[idx + 2],
            "accession":   parts[idx + 3],
        }
    except (ValueError, IndexError):
        return {"ticker": "unknown", "filing_type": "unknown", "accession": "unknown"}


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.chunks, self._buf, self._skip = [], [], False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "head"):
            self._skip = True

    def handle_endtag(self, tag):
        if tag in ("script", "style", "head"):
            self._skip = False
        if tag in ("p", "div", "tr", "li", "h1", "h2", "h3", "h4"):
            chunk = " ".join(self._buf).strip()
            if len(chunk) > 40:
                self.chunks.append(chunk)
            self._buf = []

    def handle_data(self, data):
        if not self._skip:
            self._buf.append(data.strip())


def extract_html(path):
    meta = _meta_from_path(path)
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            raw = f.read()
        extractor = _TextExtractor()
        extractor.feed(raw)
        pages = []
        for i, chunk in enumerate(extractor.chunks):
            cleaned = _clean(chunk)
            if cleaned:
                pages.append({**meta, "page": i, "text": cleaned, "source": str(path)})
        log.info(f"  HTML: {path.name} → {len(pages)} segments")
        return pages
    except Exception as e:
        log.error(f"  Failed {path.name}: {e}")
        return []


def extract_pdf(path):
    meta = _meta_from_path(path)
    pages = []
    try:
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages, 1):
                text = page.extract_text() or ""
                cleaned = _clean(text)
                if cleaned:
                    pages.append({**meta, "page": i, "text": cleaned, "source": str(path)})
        log.info(f"  PDF: {path.name} → {len(pages)} pages")
    except Exception as e:
        log.error(f"  PDF failed {path.name}: {e}")
    return pages


def parse_all():
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    all_pages = []

    files = (list(DATA_RAW.rglob("*.htm")) +
         list(DATA_RAW.rglob("*.html")) +
         list(DATA_RAW.rglob("*.pdf")) +
         list(DATA_RAW.rglob("*.txt")))

    if not files:
        log.warning("No files found. Run downloader first.")
        return []

    log.info(f"Parsing {len(files)} file(s) ...")
    for path in files:
        if path.suffix == ".pdf":
            all_pages.extend(extract_pdf(path))
        else:
            all_pages.extend(extract_html(path))

    out = DATA_PROCESSED / "parsed_pages.json"
    with open(out, "w") as f:
        json.dump(all_pages, f, indent=2)

    log.info(f"Total segments: {len(all_pages)}")
    log.info(f"Saved → {out}")
    return all_pages


if __name__ == "__main__":
    parse_all()