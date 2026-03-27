"""
Omni-Memory: Production-grade, end-to-end trainable memory system.

Built on Memory Sparse Attention (MSA) — the first latent KV-cache architecture
proven to 100M tokens with <9% degradation.
"""

__version__ = "0.1.0"

# Lazy imports to avoid requiring all dependencies at import time.
# Core config is always available:
from .config import MSAConfig

__all__ = [
    "MSAConfig",
    "encode_corpus",
    "chunk_mean_pool",
    "route_query",
    "memory_interleave",
    "TieredMemoryBank",
    "auxiliary_routing_loss",
    "combined_loss",
    "curriculum_training_schedule",
]


def __getattr__(name: str):
    """Lazy-load heavy modules only when accessed."""
    if name == "encode_corpus":
        from .encode import encode_corpus
        return encode_corpus
    if name == "chunk_mean_pool":
        from .encode import chunk_mean_pool
        return chunk_mean_pool
    if name == "route_query":
        from .route import route_query
        return route_query
    if name == "memory_interleave":
        from .interleave import memory_interleave
        return memory_interleave
    if name == "TieredMemoryBank":
        from .parallel import TieredMemoryBank
        return TieredMemoryBank
    if name == "auxiliary_routing_loss":
        from .train import auxiliary_routing_loss
        return auxiliary_routing_loss
    if name == "combined_loss":
        from .train import combined_loss
        return combined_loss
    if name == "curriculum_training_schedule":
        from .train import curriculum_training_schedule
        return curriculum_training_schedule
    raise AttributeError(f"module 'msa_memory' has no attribute {name!r}")
