"""
Tests for MSAConfig and core configuration behaviour.
"""

from msa_memory.config import MSAConfig


class TestMSAConfig:
    """Test suite for MSAConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values match paper recommendations."""
        config = MSAConfig()
        assert config.backbone_model == "Qwen/Qwen3-4B-Instruct"
        assert config.chunk_size == 64
        assert config.top_k_docs == 16
        assert config.router_layers_fraction == 0.5
        assert config.head_dim == 128
        assert config.num_heads == 8

    def test_scale_defaults(self):
        """Test scale limits are set correctly."""
        config = MSAConfig()
        assert config.max_memory_tokens == 100_000_000
        assert config.max_query_tokens == 4096
        assert config.max_answer_tokens == 1024

    def test_training_defaults(self):
        """Test training hyperparameters match paper values."""
        config = MSAConfig()
        assert config.warmup_lm_weight == 0.1
        assert config.warmup_aux_weight == 1.0
        assert config.main_lm_weight == 1.0
        assert config.main_aux_weight == 0.1
        assert config.temperature == 0.07

    def test_custom_values(self):
        """Test custom configuration overrides."""
        config = MSAConfig(
            backbone_model="meta-llama/Llama-3-8B-Instruct",
            chunk_size=128,
            top_k_docs=32,
            max_memory_tokens=10_000_000,
        )
        assert config.backbone_model == "meta-llama/Llama-3-8B-Instruct"
        assert config.chunk_size == 128
        assert config.top_k_docs == 32
        assert config.max_memory_tokens == 10_000_000

    def test_tiered_storage_defaults(self):
        """Test tiered storage defaults for Memory Parallel."""
        config = MSAConfig()
        assert config.routing_keys_device == "cuda"
        assert config.content_kvs_device == "cpu"
        assert config.async_prefetch is True

    def test_interleave_defaults(self):
        """Test Memory Interleave defaults."""
        config = MSAConfig()
        assert config.max_interleave_rounds == 5
        assert config.interleave_delimiter == "<end_of_retrieve>"
