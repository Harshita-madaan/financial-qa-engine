"""
src/downloader.py — fetch SEC filings from EDGAR.
Run: python -m src.downloader
"""
import logging
from pathlib import Path
from sec_edgar_downloader import Downloader
from config import DATA_RAW, TICKERS, FILING_TYPES, NUM_FILINGS

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def download_filings():
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    dl = Downloader("FinancialQA", "user@example.com", DATA_RAW)
    downloaded = []

    for ticker in TICKERS:
        for filing_type in FILING_TYPES:
            log.info(f"Downloading {NUM_FILINGS}x {filing_type} for {ticker} ...")
            try:
                dl.get(filing_type, ticker, limit=NUM_FILINGS)
                ticker_dir = DATA_RAW / "sec-edgar-filings" / ticker / filing_type
                if ticker_dir.exists():
                    paths = (list(ticker_dir.rglob("*.htm")) +
                             list(ticker_dir.rglob("*.html")) +
                             list(ticker_dir.rglob("*.txt")))
                    downloaded.extend(paths)
                    log.info(f"  → {len(paths)} file(s) saved")
            except Exception as e:
                log.warning(f"  ✗ Failed {ticker}/{filing_type}: {e}")

    log.info(f"Total files downloaded: {len(downloaded)}")
    return downloaded


if __name__ == "__main__":
    download_filings()