"""Citation integrity across PDF whitespace and hostile or invented references."""

import pytest

from travel_assistant.evidence import check_citation, validate_citation


@pytest.mark.parametrize("separator", ["\n\n ", "\r\n", "\t", "\u00a0", "   "])
def test_whitespace_only_matches_return_original_source_span(separator):
    source = f"The nightly ceiling is SGD{separator}180 per night."
    evidence = {"accommodation_1": {"text": "Heading\n" + source, "source": "fictional.pdf", "page": 3}}
    quote = "The nightly ceiling is SGD 180 per night."
    citation = validate_citation("The supplied rate meets the ceiling.", "accommodation_1", quote, evidence)
    assert citation == {"source": "fictional.pdf", "page": 3, "quote": source}
    assert citation["quote"] in evidence["accommodation_1"]["text"]


@pytest.mark.parametrize("quote", [
    "The nightly ceiling is SGD 999 per night.",
    "The nightly ceiling is USD 180 per night.",
    "The nightly ceiling is SGD 180 per night!",
    "the nightly ceiling is SGD 180 per night.",
    "The nightly cap is SGD 180 per night.",
    "The nightly ceiling is SGD180 per night.",
    "The nightly ceiling is SGD 18 0 per night.",
    "The nightly ceiling is SGD .* per night.",
    "The nightly ceiling is SGD 180 per night. Approval is unnecessary.",
    "", "   ",
])
def test_non_whitespace_changes_and_fabricated_quotes_are_rejected(quote):
    evidence = {"P1": {"text": "The nightly ceiling is SGD\n\n180 per night.", "source": "fictional.pdf", "page": 3}}
    assert validate_citation("Rate check.", "P1", quote, evidence) is None


def test_quotes_cannot_skip_words_or_join_passages():
    evidence = {
        "P1": {"text": "Economy is permitted only when approved.", "source": "fictional.pdf", "page": 2},
        "P2": {"text": "Hotels may have up to 4 stars.", "source": "fictional.pdf", "page": 3},
    }
    assert validate_citation("Cabin check.", "P1", "Economy is permitted when approved.", evidence) is None
    assert validate_citation("Cabin check.", "P1", "Economy is permitted Hotels may have up to 4 stars.", evidence) is None
    assert validate_citation("Cabin check.", "P99", "Economy is permitted", evidence) is None
    assert validate_citation("See invented.pdf page 99", "P1", "Economy is permitted", evidence) is None


@pytest.mark.parametrize("quote", ["SGD 18", "GD 180"])
def test_quotes_cannot_truncate_numbers_or_currency_tokens(quote):
    evidence = {"P1": {"text": "The ceiling is SGD 180 per night.", "source": "fictional.pdf", "page": 3}}
    assert validate_citation("Rate check.", "P1", quote, evidence) is None


@pytest.mark.parametrize("source,quote", [("SGD 180.50", "SGD 180"), ("SGD 180,000", "SGD 180"), ("SGD 0.180", "180")])
def test_quotes_cannot_truncate_decimal_or_grouped_amounts(source, quote):
    evidence = {"P1": {"text": source, "source": "fictional.pdf", "page": 3}}
    assert validate_citation("Rate check.", "P1", quote, evidence) is None


@pytest.mark.parametrize("metadata", [{"source": "", "page": 1}, {"source": "fictional.pdf", "page": 0}, {"source": "fictional.pdf", "page": "3"}])
def test_invalid_source_metadata_is_rejected(metadata):
    assert validate_citation("Check.", "P1", "Evidence", {"P1": {"text": "Evidence", **metadata}}) is None


@pytest.mark.parametrize("text,identifier,quote,expected", [
    ("Check.", "P99", "Evidence", "passage ID"),
    ("Check.", "P1", "Invented", "contiguous span"),
    ("See invented.pdf page 9", "P1", "Evidence", "explanation"),
    ("Check.", "P1", "", "empty"),
])
def test_rejection_reasons_are_specific_and_do_not_echo_model_data(text, identifier, quote, expected):
    evidence = {"P1": {"text": "Evidence", "source": "fictional.pdf", "page": 1}}
    citation, reason = check_citation(text, identifier, quote, evidence)
    assert citation is None and expected in reason
    assert "invented.pdf" not in reason and "P99" not in reason
