"""
Tests for Memory Interleave helper functions.
"""

from msa_memory.interleave import extract_doc_ids, clean_answer


class TestExtractDocIds:
    """Test suite for parsing document IDs from generated text."""

    def test_standard_format(self):
        """Parse standard [id] format."""
        text = "[4] [7] [12] <end_of_retrieve>"
        ids = extract_doc_ids(text)
        assert ids == [4, 7, 12]

    def test_no_ids(self):
        """Handle text with no document IDs."""
        text = "The answer is 42. <end_of_retrieve>"
        ids = extract_doc_ids(text)
        assert ids == []

    def test_single_id(self):
        """Parse a single document ID."""
        text = "[99] <end_of_retrieve>"
        ids = extract_doc_ids(text)
        assert ids == [99]

    def test_ids_in_prose(self):
        """Parse IDs embedded in prose text."""
        text = "I need document [3] and also [15] for context <end_of_retrieve>"
        ids = extract_doc_ids(text)
        assert ids == [3, 15]


class TestCleanAnswer:
    """Test suite for cleaning generated answer text."""

    def test_remove_delimiter(self):
        """Remove the interleave delimiter from answer."""
        text = "The answer is 42.<end_of_retrieve>"
        assert "<end_of_retrieve>" not in clean_answer(text)

    def test_remove_doc_ids(self):
        """Remove document ID references from answer."""
        text = "Based on [4] and [7], the answer is 42."
        cleaned = clean_answer(text)
        assert "[4]" not in cleaned
        assert "[7]" not in cleaned
        assert "42" in cleaned

    def test_strip_whitespace(self):
        """Clean should strip leading/trailing whitespace."""
        text = "  the answer  "
        assert clean_answer(text) == "the answer"
