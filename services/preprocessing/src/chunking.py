import re
import logging
from typing import List, Optional, Sequence

try:
    import tiktoken
except ImportError:  # pragma: no cover - optional dependency
    tiktoken = None


NUMBERED_LINE_RE = re.compile(r"^\s*(?:\[(\d+)\]|(\d+)\.)\s+(.*)")
SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[A-Z0-9])")
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)

logger = logging.getLogger(__name__)

_ENCODER = None
if tiktoken is not None:  # pragma: no cover - optional dependency
    try:
        _ENCODER = tiktoken.get_encoding("cl100k_base")
    except Exception:
        _ENCODER = None


def trim_document_header(text: str) -> str:
    """
    Remove top-of-document boilerplate before sentence chunking.
    This intentionally treats heading blocks such as ORDER/JUDGMENT plus the judge's
    sign-off line as part of the header and strips them before chunk creation.
    """
    lines = text.splitlines()
    if not lines:
        return text.strip()

    start_index = None
    saw_heading = False

    for index, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line:
            continue

        if _is_numbered_paragraph(line):
            start_index = index
            break

        if _is_header_line(line) or _is_judge_line(line) or _looks_like_citation_block(line):
            saw_heading = True
            continue

        if saw_heading and _looks_like_body_line(line):
            start_index = index
            break

    if start_index is None:
        fallback_index = _first_likely_body_index(lines)
        if fallback_index is not None:
            logger.warning(
                "Header trim could not confidently find a boundary; falling back to line %s.",
                fallback_index + 1,
            )
            return "\n".join(lines[fallback_index:]).strip()
        logger.warning("Header trim could not find any body boundary; returning original text unchanged.")
        return text.strip()

    return "\n".join(lines[start_index:]).strip()


def build_chunks(
    text: str,
    target_tokens: int = 500,
    max_tokens: int = 600,
    min_tokens: int = 400,
    overlap_sentences: int = 2,
) -> List[dict]:
    """
    Create stable sentence-aware chunks with small overlap.
    """
    sentences = expand_long_sentences(split_into_sentences(text), max_tokens=max_tokens)
    if not sentences:
        return []

    chunks: List[dict] = []
    start = 0
    while start < len(sentences):
        end = start
        chunk_sentences: List[str] = []
        chunk_tokens = 0

        while end < len(sentences):
            sentence = sentences[end]
            sentence_tokens = estimate_tokens(sentence)
            if chunk_sentences and chunk_tokens + sentence_tokens > max_tokens and chunk_tokens >= min_tokens:
                break

            chunk_sentences.append(sentence)
            chunk_tokens += sentence_tokens
            end += 1
            if chunk_tokens >= target_tokens:
                break

        if not chunk_sentences and end < len(sentences):
            chunk_sentences = [sentences[end]]
            end += 1

        if not chunk_sentences:
            break

        chunks.append(
            {
                "cid": len(chunks) + 1,
                "text": " ".join(chunk_sentences).strip(),
            }
        )

        if end >= len(sentences):
            break

        next_start = max(end - min(overlap_sentences, len(chunk_sentences)), start + 1)
        start = next_start

    return chunks


def split_into_sentences(text: str) -> List[str]:
    sentences: List[str] = []
    for block in _iter_body_blocks(text):
        normalized = re.sub(r"\s*\n+\s*", " ", block).strip()
        if not normalized:
            continue
        protected = _protect_abbreviations(normalized)
        parts = SENTENCE_BOUNDARY_RE.split(protected)
        for part in parts:
            sentence = _unprotect_abbreviations(part).strip()
            if sentence:
                sentences.append(sentence)
    return sentences


def estimate_tokens(text: str) -> int:
    if _ENCODER is not None:  # pragma: no cover - optional dependency
        return len(_ENCODER.encode(text))
    return len(TOKEN_RE.findall(text))


def split_long_sentence(sentence: str, max_tokens: int = 600) -> List[str]:
    if estimate_tokens(sentence) <= max_tokens:
        return [sentence.strip()]

    for pattern in (r"(?<=[;:])\s+", r"(?<=[,])\s+"):
        pieces = [piece.strip() for piece in re.split(pattern, sentence) if piece.strip()]
        if len(pieces) > 1:
            fragments: List[str] = []
            for piece in pieces:
                fragments.extend(split_long_sentence(piece, max_tokens=max_tokens))
            return fragments

    words = sentence.split()
    if len(words) <= 1:
        return [sentence.strip()]

    fragments: List[str] = []
    buffer: List[str] = []
    buffer_tokens = 0
    for word in words:
        word_tokens = estimate_tokens(word)
        if buffer and buffer_tokens + word_tokens > max_tokens:
            fragments.append(" ".join(buffer).strip())
            buffer = [word]
            buffer_tokens = word_tokens
            continue
        buffer.append(word)
        buffer_tokens += word_tokens

    if buffer:
        fragments.append(" ".join(buffer).strip())

    return [fragment for fragment in fragments if fragment]


def expand_long_sentences(sentences: Sequence[str], max_tokens: int = 600) -> List[str]:
    expanded: List[str] = []
    for sentence in sentences:
        if estimate_tokens(sentence) <= max_tokens:
            expanded.append(sentence)
            continue
        expanded.extend(split_long_sentence(sentence, max_tokens=max_tokens))
    return expanded


def _iter_body_blocks(text: str) -> List[str]:
    blocks: List[str] = []
    current: List[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if current:
                blocks.append("\n".join(current).strip())
                current = []
            continue

        if _is_numbered_paragraph(line) and current:
            blocks.append("\n".join(current).strip())
            current = [_strip_number_marker(line)]
            continue

        if _is_numbered_paragraph(line):
            current = [_strip_number_marker(line)]
            continue

        current.append(line)

    if current:
        blocks.append("\n".join(current).strip())

    return blocks


def _is_numbered_paragraph(line: str) -> bool:
    return bool(NUMBERED_LINE_RE.match(line))


def _strip_number_marker(line: str) -> str:
    match = NUMBERED_LINE_RE.match(line)
    if not match:
        return line.strip()
    return match.group(3).strip()


def _is_header_line(line: str) -> bool:
    compact = re.sub(r"\s+", " ", line).strip()
    if not compact:
        return True
    if re.fullmatch(r"(?i)(?:ORDER|JUDGMENT|J U D G M E N T|REPORTABLE|NON-REPORTABLE)(?:[:\s].*)?", compact):
        return True
    if re.fullmatch(r"(?i)(?:AUTHOR|BENCH):.*", compact):
        return True
    if re.fullmatch(r"(?i)(?:THE\s+)?SUPREME\s+COURT.*", compact):
        return True
    if re.fullmatch(r"(?i)(?:CIVIL|CRIMINAL|ORIGINAL)\s+APPELLATE.*", compact):
        return True
    if re.fullmatch(r"(?i)EQUIVALENT\s+CITATIONS:.*", compact):
        return True
    return False


def _is_judge_line(line: str) -> bool:
    compact = re.sub(r"\s+", " ", line).strip()
    return bool(
        re.fullmatch(
            r"(?i)(?:DR\.\s*)?[A-Z][A-Z.\s]*,\s*J\.?",
            compact,
        )
    )


def _looks_like_citation_block(line: str) -> bool:
    compact = re.sub(r"\s+", " ", line).strip()
    return bool(
        re.search(r"\b\d{4}\b", compact)
        and (
            "citations" in compact.lower()
            or "air" in compact.lower()
            or "scc" in compact.lower()
            or "jt" in compact.lower()
            or "scw" in compact.lower()
        )
    )


def _looks_like_body_line(line: str) -> bool:
    compact = re.sub(r"\s+", " ", line).strip()
    if len(compact) < 12:
        return False
    if compact.endswith(":"):
        return True
    if re.match(r"(?i)^(the|in|we|it|this|that|however|thus|hence|after|before|when|where|whether|while|therefore)\b", compact):
        return True
    return bool(re.search(r"[a-z]", compact))


def _first_likely_body_index(lines: Sequence[str]) -> Optional[int]:
    for index, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line:
            continue
        if _is_numbered_paragraph(line) or _is_clear_body_line(line):
            return index
    return None


def _is_clear_body_line(line: str) -> bool:
    compact = re.sub(r"\s+", " ", line).strip()
    if len(compact) < 12:
        return False
    if _is_numbered_paragraph(compact):
        return True
    return bool(
        compact.endswith(":")
        or re.match(
            r"(?i)^(the|in|we|it|this|that|however|thus|hence|after|before|when|where|whether|while|therefore)\b",
            compact,
        )
    )


def _protect_abbreviations(text: str) -> str:
    protected = text
    for token in (
        "Mr.",
        "Mrs.",
        "Ms.",
        "Dr.",
        "Prof.",
        "Shri.",
        "Smt.",
        "Sri.",
        "No.",
        "Art.",
        "Sec.",
        "e.g.",
        "i.e.",
        "vs.",
        "v.",
    ):
        protected = protected.replace(token, token.replace(".", "<DOT>"))

    protected = re.sub(r"\b(?:[A-Z]\.){2,}", lambda match: match.group(0).replace(".", "<DOT>"), protected)
    return protected


def _unprotect_abbreviations(text: str) -> str:
    return text.replace("<DOT>", ".")
