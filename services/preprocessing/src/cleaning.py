import re
from collections import Counter
from typing import List, Sequence


HEADER_TAIL_WINDOW = 4
FOOTER_TAIL_WINDOW = 4


def normalize_line(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_lines(page_text: str) -> List[str]:
    return [normalize_line(line) for line in page_text.splitlines() if normalize_line(line)]


def _is_generic_noise_line(line: str) -> bool:
    lower = line.lower()
    if not lower:
        return True
    if "indian kanoon" in lower:
        return True
    if "reportable" in lower:
        return True
    if re.fullmatch(r"(?:page\s*)?\d+(?:\s*(?:of|/)\s*\d+)?", lower):
        return True
    if re.fullmatch(r"[-–—\s]*\d+[-–—\s]*", lower):
        return True
    return False


def _edge_candidates(pages: Sequence[List[str]]) -> tuple[set[str], set[str]]:
    if not pages:
        return set(), set()

    header_counts: Counter[str] = Counter()
    footer_counts: Counter[str] = Counter()
    for lines in pages:
        if not lines:
            continue
        for line in lines[:HEADER_TAIL_WINDOW]:
            header_counts[line] += 1
        for line in lines[-FOOTER_TAIL_WINDOW:]:
            footer_counts[line] += 1

    threshold = 2 if len(pages) > 1 else 10**9
    header_candidates = {line for line, count in header_counts.items() if count >= threshold}
    footer_candidates = {line for line, count in footer_counts.items() if count >= threshold}
    return header_candidates, footer_candidates


def clean_pages(page_texts: Sequence[str]) -> List[str]:
    """
    Remove repeated page headers, footers, and obvious scanner noise.
    """
    pages = [split_lines(page_text) for page_text in page_texts]
    header_candidates, footer_candidates = _edge_candidates(pages)
    cleaned_pages: List[str] = []

    for lines in pages:
        cleaned_lines: List[str] = []
        last_index = len(lines) - 1
        for index, line in enumerate(lines):
            if _is_generic_noise_line(line):
                continue
            if line in footer_candidates and index >= max(0, last_index - FOOTER_TAIL_WINDOW + 1):
                continue
            if line in header_candidates and index < HEADER_TAIL_WINDOW:
                continue
            cleaned_lines.append(line)
        cleaned_pages.append("\n".join(cleaned_lines).strip())

    return cleaned_pages


def normalize_document_text(page_texts: Sequence[str]) -> str:
    text = "\n\n".join(page_texts).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
