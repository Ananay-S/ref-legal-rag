import logging
from typing import List

logger = logging.getLogger(__name__)

import fitz


def extract_pages(pdf_path: str) -> List[str]:
    """
    Extract raw text from each PDF page in order.
    """
    logger.info("Extracting pages with fitz: %s", pdf_path)
    doc = fitz.open(pdf_path)
    try:
        pages: List[str] = []
        for page_index, page in enumerate(doc, start=1):
            text = page.get_text("text") or ""
            if not text.strip():
                logger.warning("No text extracted from page %s of %s", page_index, pdf_path)
            pages.append(text)
        return pages
    finally:
        doc.close()
