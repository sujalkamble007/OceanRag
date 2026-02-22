"""
document_loader.py — PDF loading + metadata extraction using LangChain.
"""

import os
import signal
from pathlib import Path
from tqdm import tqdm
from langchain_community.document_loaders import PyPDFLoader


class PDFTimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise PDFTimeoutError("PDF loading timed out")


def load_documents(docs_dir: str, timeout_per_pdf: int = 60) -> list:
    """
    Scans docs_dir recursively for .pdf files, loads each with PyPDFLoader,
    and enriches metadata with filename, filepath, and 1-indexed page_number.
    Skips PDFs that take longer than timeout_per_pdf seconds to parse.
    Returns a list of LangChain Document objects.
    """
    docs_path = Path(docs_dir)
    pdf_files = sorted(docs_path.rglob("*.pdf"))

    if not pdf_files:
        print(f"⚠️  No PDF files found in '{docs_dir}'")
        return []

    print(f"📂 Found {len(pdf_files)} PDF(s). Loading...")

    all_docs = []
    skipped = []
    for pdf_file in tqdm(pdf_files, desc="Loading PDFs"):
        try:
            # Set a timeout to skip PDFs that hang
            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(timeout_per_pdf)

            loader = PyPDFLoader(str(pdf_file))
            pages = loader.load()

            signal.alarm(0)  # Cancel the alarm

            for page in pages:
                page.metadata["filename"] = pdf_file.name
                page.metadata["filepath"] = str(pdf_file)
                # PyPDFLoader uses 0-indexed 'page'; convert to 1-indexed
                page.metadata["page_number"] = page.metadata.get("page", 0) + 1

            all_docs.extend(pages)
        except PDFTimeoutError:
            signal.alarm(0)
            skipped.append(pdf_file.name)
            print(f"\n⏰ Skipping '{pdf_file.name}': timed out after {timeout_per_pdf}s")
        except Exception as e:
            signal.alarm(0)
            skipped.append(pdf_file.name)
            print(f"\n⚠️  Skipping '{pdf_file.name}': {e}")

    if skipped:
        print(f"⚠️  Skipped {len(skipped)} problematic PDF(s): {', '.join(skipped)}")
    print(f"✅ Loaded {len(all_docs)} pages from {len(pdf_files) - len(skipped)} document(s).")
    return all_docs


def summarize_documents(docs: list) -> dict:
    """
    Prints a formatted table of filename → page count.
    Returns dict mapping filename → page_count and includes total page count.
    """
    file_pages = {}
    for doc in docs:
        fname = doc.metadata.get("filename", "unknown")
        if fname not in file_pages:
            file_pages[fname] = 0
        file_pages[fname] += 1

    print("\n📄 Document Summary:")
    print(f"{'Filename':<55} {'Pages':>5}")
    print("-" * 62)
    for fname, count in file_pages.items():
        print(f"{fname:<55} {count:>5}")
    print("-" * 62)
    print(f"{'TOTAL':<55} {len(docs):>5}")
    print()

    return file_pages
