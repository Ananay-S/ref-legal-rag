import re
import logging
from typing import List, Sequence

from cleaning import normalize_line, split_lines

logger = logging.getLogger(__name__)

HEADER_WINDOW_NON_EMPTY_LINES = 15


def extract_metadata(page_texts: Sequence[str]) -> dict:
    """
    Parse document-level metadata from the first page and header block.
    """
    first_page = page_texts[0] if page_texts else ""
    lines = split_lines(first_page)
    header_lines = _bounded_header_lines(lines)
    appellant = "Not Found"
    respondent = "Not Found"
    day = 1
    month = 1
    year = 2000

    case_name_parts: List[str] = []
    for line in header_lines:
        if _is_metadata_noise_line(line):
            continue
        if _is_body_heading(line):
            break

        date_match = re.search(r"(?i)\bon\s+(\d{1,2})\s+([A-Za-z]+),?\s*(\d{4})\b", line)
        if date_match:
            left_side = line[:date_match.start()].strip()
            if left_side:
                case_name_parts.append(left_side)
            day = int(date_match.group(1))
            month = _month_number(date_match.group(2))
            year = int(date_match.group(3))
            break

        case_name_parts.append(line)

    case_name = normalize_line(" ".join(case_name_parts))
    vs_split = re.split(r"(?i)\s+(?:vs|v\.|versus)\s+", case_name, maxsplit=1)
    if len(vs_split) == 2:
        appellant = vs_split[0].strip()
        respondent = vs_split[1].strip()
    elif case_name and case_name != "Not Found":
        logger.debug("Could not split case name into appellant/respondent: %s", case_name)

    author = "Not Found"
    bench: List[str] = []
    for line in header_lines:
        author_match = re.search(r"(?i)^\s*Author:\s*(.+)$", line)
        if author_match:
            author = author_match.group(1).strip()
            continue
        bench_match = re.search(r"(?i)^\s*Bench:\s*(.+)$", line)
        if bench_match:
            bench = [item.strip() for item in bench_match.group(1).split(",") if item.strip()]

    jurisdiction = "Not Found"
    jurisdiction_match = re.search(r"(?i)\b(CIVIL APPELLATE|CRIMINAL APPELLATE|ORIGINAL)\b", first_page)
    if jurisdiction_match:
        jurisdiction = jurisdiction_match.group(1).title()

    return {
        "appellant": appellant,
        "respondent": respondent,
        "year": year,
        "month": month,
        "day": day,
        "author": author,
        "bench": bench,
        "court": "Supreme Court of India",
        "jurisdiction": jurisdiction,
    }


def _is_metadata_noise_line(line: str) -> bool:
    compact = re.sub(r"\s+", " ", line).strip()
    return bool(
        re.fullmatch(r"(?i)EQUIVALENT\s+CITATIONS:.*", compact)
        or re.fullmatch(r"(?i)(?:AUTHOR|BENCH):.*", compact)
        or re.fullmatch(r"(?i)(?:IN\s+)?(?:THE\s+)?SUPREME\s+COURT.*", compact)
        or re.fullmatch(r"(?i)(?:CIVIL|CRIMINAL|ORIGINAL)\s+APPELLATE.*", compact)
        or re.fullmatch(r"(?i)(?:REPORTABLE|NON-REPORTABLE)\b.*", compact)
        or re.fullmatch(r"(?i)(?:JUDGMENT|ORDER|J U D G M E N T)\b.*", compact)
        or re.fullmatch(r"(?i).*\b(?:ORDER|JUDGMENT)\s+NO\.?.*", compact)
    )


def _bounded_header_lines(lines: Sequence[str], limit: int = HEADER_WINDOW_NON_EMPTY_LINES) -> List[str]:
    bounded: List[str] = []
    non_empty_seen = 0
    for line in lines:
        if line.strip():
            non_empty_seen += 1
        bounded.append(line)
        if non_empty_seen >= limit:
            break
    return bounded


def _is_body_heading(line: str) -> bool:
    compact = re.sub(r"\s+", " ", line).strip()
    return bool(
        re.fullmatch(r"(?i)(?:ORDER|JUDGMENT|J U D G M E N T|REPORTABLE|NON-REPORTABLE)(?:[:\s].*)?", compact)
        or re.fullmatch(r"(?i)(?:AUTHOR|BENCH):.*", compact)
    )


def _month_number(month_name: str) -> int:
    month_lookup = {
        "january": 1,
        "jan": 1,
        "february": 2,
        "feb": 2,
        "march": 3,
        "mar": 3,
        "april": 4,
        "apr": 4,
        "may": 5,
        "june": 6,
        "jun": 6,
        "july": 7,
        "jul": 7,
        "august": 8,
        "aug": 8,
        "september": 9,
        "sep": 9,
        "sept": 9,
        "october": 10,
        "oct": 10,
        "november": 11,
        "nov": 11,
        "december": 12,
        "dec": 12,
    }
    return month_lookup.get(month_name.lower(), 1)
