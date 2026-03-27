"""
Tests for the chunk_mean_pool function and encoding utilities.
"""

import torch
import pytest

from msa_memory.encode import chunk_mean_pool


class TestChunkMeanPool:
    """Test suite for the φ(·) chunk-wise mean pooling function."""

    def test_exact_divisible(self):
        """Test pooling when seq_len is exactly divisible by chunk_size."""
        # [1, 128, 64] with chunk_size=64 → [1, 2, 64]
        tensor = torch.randn(1, 128, 64)
        result = chunk_mean_pool(tensor, chunk_size=64)
        assert result.shape == (1, 2, 64)

    def test_with_padding(self):
        """Test pooling when seq_len requires padding."""
        # [1, 100, 32] with chunk_size=64 → needs padding to 128 → [1, 2, 32]
        tensor = torch.randn(1, 100, 32)
        result = chunk_mean_pool(tensor, chunk_size=64)
        assert result.shape == (1, 2, 32)

    def test_single_chunk(self):
        """Test pooling with seq_len <= chunk_size."""
        tensor = torch.randn(1, 30, 64)
        result = chunk_mean_pool(tensor, chunk_size=64)
        assert result.shape == (1, 1, 64)

    def test_mean_correctness(self):
        """Test that pooling actually computes the mean."""
        # Create a known tensor: first chunk all 1s, second chunk all 2s
        tensor = torch.ones(1, 4, 2)
        tensor[:, 2:, :] = 2.0  # second half = 2.0

        result = chunk_mean_pool(tensor, chunk_size=2)
        assert result.shape == (1, 2, 2)
        assert torch.allclose(result[0, 0], torch.tensor([1.0, 1.0]))
        assert torch.allclose(result[0, 1], torch.tensor([2.0, 2.0]))

    def test_compression_ratio(self):
        """Test that compression ratio matches chunk_size."""
        seq_len = 1024
        chunk_size = 64
        tensor = torch.randn(1, seq_len, 128)
        result = chunk_mean_pool(tensor, chunk_size)
        # 1024 / 64 = 16 chunks
        assert result.shape[1] == seq_len // chunk_size

    def test_batch_dimension_preserved(self):
        """Test that batch dimension is preserved."""
        tensor = torch.randn(4, 256, 64)
        result = chunk_mean_pool(tensor, chunk_size=64)
        assert result.shape[0] == 4
