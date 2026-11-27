"""Shared validation of model claims against retrieved policy passages."""

import re


def check_citation(text: str, passage_id: str, quote: str, evidence: dict) -> tuple[dict | None, str | None]:
    """Match a contiguous source span, allowing differences in whitespace only.

    PDF line wrapping is presentation, while words, numbers, case and punctuation
    must match exactly. Return the original source span for display and checking.
    """
    passage = evidence.get(passage_id)
    if passage is None:
        return None, "The supporting passage ID was not retrieved for this section."
    if (not isinstance(text, str) or not text or len(text) > 1000
            or re.search(r"\bpages?\s+\d|\.pdf\b|\[\d+\]", text, re.IGNORECASE)):
        return None, "The generated explanation is unusable or contains unsupported citation text."
    if not isinstance(quote, str) or not quote.strip():
        return None, "The supporting quote is empty or unusable."
    if (not isinstance(passage.get("source"), str) or not passage["source"].strip()
            or type(passage.get("page")) is not int or passage["page"] < 1):
        return None, "The retrieved passage has invalid filename or page metadata."
    # Escaping tokens prevents model-generated quote text becoming a regex.
    quote = quote.strip()
    prefix = r"(?<![\w.,])" if quote[0].isdigit() else r"(?<!\w)"
    suffix = r"(?!\w|[.,]\d)" if quote[-1].isdigit() else r"(?!\w)"
    pattern = prefix + r"\s+".join(re.escape(token) for token in quote.split()) + suffix
    match = re.search(pattern, passage["text"])
    if match is None:
        return None, "The supporting quote does not match a contiguous span of the cited passage."
    return {"source": passage["source"], "page": passage["page"], "quote": match.group()}, None


def validate_citation(text: str, passage_id: str, quote: str, evidence: dict) -> dict | None:
    """Preserve the existing Q&A interface using the same evidence checks."""
    citation, _ = check_citation(text, passage_id, quote, evidence)
    return citation
