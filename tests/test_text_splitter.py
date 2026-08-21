"""
Regression tests for the chunking cascade in app/utils/text_splitter.py.

Covers two defects:
  1. split_by_section dropped all text preceding the first section header,
     so preambles and definitions clauses were never embedded.
  2. split_text looped forever when chunk_overlap >= chunk_size, because the
     start pointer advances by (chunk_size - chunk_overlap).

Only the regex and fixed-width tiers are exercised here; semantic_split pulls
in the spaCy model, which is too slow for a unit test.
"""
import pytest

from app.utils.text_splitter import split_by_section, split_text


LEGAL_TEXT = """Preamble: This Act may be called the Digital Personal Data Protection Act.

Section 4 Grounds for processing. A Data Fiduciary may process personal data
only for a lawful purpose for which the Data Principal has given consent.

Section 5 Notice. Every request for consent shall be accompanied by a notice
informing the Data Principal of the purpose of processing.
"""


def test_preamble_is_preserved():
    """Text before the first header must survive — it used to be dropped."""
    chunks = split_by_section(LEGAL_TEXT)

    all_text = " ".join(chunk["text"] for chunk in chunks)
    assert "Preamble" in all_text
    assert "Digital Personal Data Protection Act" in all_text


def test_preamble_is_labelled():
    chunks = split_by_section(LEGAL_TEXT)

    assert chunks[0]["metadata"]["section_title"] == "Preamble"


def test_sections_are_split_and_labelled():
    chunks = split_by_section(LEGAL_TEXT)

    titles = [chunk["metadata"]["section_title"] for chunk in chunks]
    assert "Section 4" in titles
    assert "Section 5" in titles


def test_no_preamble_chunk_when_text_starts_with_a_header():
    text = "Section 1 Short title. This Act may be called the DPDP Act."

    chunks = split_by_section(text)

    assert len(chunks) == 1
    assert chunks[0]["metadata"]["section_title"] == "Section 1"


def test_unstructured_text_yields_no_sections():
    """Callers rely on an empty list to fall through to the next tier."""
    assert split_by_section("Just a paragraph with no headers at all.") == []


def test_fixed_split_overlaps_by_the_configured_amount():
    words = " ".join(f"w{i}" for i in range(20))

    chunks = split_text(words, chunk_size=8, chunk_overlap=3)

    assert chunks[0].split()[-3:] == chunks[1].split()[:3]


def test_fixed_split_covers_all_words():
    words = " ".join(f"w{i}" for i in range(20))

    chunks = split_text(words, chunk_size=8, chunk_overlap=3)

    seen = {word for chunk in chunks for word in chunk.split()}
    assert seen == set(words.split())


@pytest.mark.parametrize("chunk_size, chunk_overlap", [(5, 5), (5, 8)])
def test_overlap_not_less_than_chunk_size_is_rejected(chunk_size, chunk_overlap):
    """Previously an infinite loop appending chunks until memory ran out."""
    with pytest.raises(ValueError, match="must be less than"):
        split_text("some text here", chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def test_non_positive_chunk_size_is_rejected():
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        split_text("some text here", chunk_size=0, chunk_overlap=0)
