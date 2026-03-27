"""
Tests for RoPE utilities (doc-wise and query position IDs).
"""

import torch

from msa_memory.rope_utils import (
    build_docwise_position_ids,
    build_query_position_ids,
    build_combined_position_ids,
)


class TestDocwisePositionIds:
    """Test suite for doc-wise RoPE position ID generation."""

    def test_single_document(self):
        """Test position IDs for a single document."""
        pos_ids = build_docwise_position_ids([100])
        assert pos_ids.shape == (1, 100)
        assert pos_ids[0, 0].item() == 0
        assert pos_ids[0, 99].item() == 99

    def test_multiple_documents_reset(self):
        """Test that position IDs reset for each document."""
        pos_ids = build_docwise_position_ids([50, 30, 20])
        assert pos_ids.shape == (1, 100)

        # First doc: 0-49
        assert pos_ids[0, 0].item() == 0
        assert pos_ids[0, 49].item() == 49

        # Second doc: reset to 0-29
        assert pos_ids[0, 50].item() == 0
        assert pos_ids[0, 79].item() == 29

        # Third doc: reset to 0-19
        assert pos_ids[0, 80].item() == 0
        assert pos_ids[0, 99].item() == 19

    def test_no_global_explosion(self):
        """Test that doc-wise IDs never exceed individual doc length."""
        doc_lengths = [100] * 1000  # 1000 documents of 100 tokens each
        pos_ids = build_docwise_position_ids(doc_lengths)

        # Max position ID should be 99 (within a single doc), NOT 99999
        assert pos_ids.max().item() == 99


class TestQueryPositionIds:
    """Test suite for query position ID generation."""

    def test_offset_by_k(self):
        """Test that query positions are offset by number of retrieved docs."""
        pos_ids = build_query_position_ids(query_len=512, num_retrieved_docs=16)
        assert pos_ids.shape == (1, 512)
        assert pos_ids[0, 0].item() == 16
        assert pos_ids[0, 511].item() == 527

    def test_zero_offset(self):
        """Test query with no retrieved documents."""
        pos_ids = build_query_position_ids(query_len=100, num_retrieved_docs=0)
        assert pos_ids[0, 0].item() == 0
        assert pos_ids[0, 99].item() == 99


class TestCombinedPositionIds:
    """Test suite for combined doc + query position IDs."""

    def test_combined_shape(self):
        """Test combined position IDs have correct total length."""
        pos_ids = build_combined_position_ids(
            doc_lengths=[100, 200],
            query_len=50,
        )
        assert pos_ids.shape == (1, 350)  # 100 + 200 + 50

    def test_combined_values(self):
        """Test docwise resets followed by query offset."""
        pos_ids = build_combined_position_ids(
            doc_lengths=[10, 10],
            query_len=5,
        )
        # Doc 1: 0-9
        assert pos_ids[0, 0].item() == 0
        assert pos_ids[0, 9].item() == 9
        # Doc 2: 0-9 (reset)
        assert pos_ids[0, 10].item() == 0
        assert pos_ids[0, 19].item() == 9
        # Query: offset by 2 (number of docs)
        assert pos_ids[0, 20].item() == 2
        assert pos_ids[0, 24].item() == 6
