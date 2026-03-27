"""
MSA Configuration — All hyperparameters in one place.

This module defines the central configuration dataclass for the entire
Omni-Memory system. All components (encode, route, generate, train, interleave,
parallel) read from this single config to ensure consistency.

Usage:
    from msa_memory.config import MSAConfig

    # Default configuration (Qwen3-4B backbone, 100M token ceiling)
    config = MSAConfig()

    # Custom configuration
    config = MSAConfig(
        backbone_model="meta-llama/Llama-3-8B-Instruct",
        chunk_size=128,
        top_k_docs=32,
        max_memory_tokens=10_000_000,  # 10M tokens (fits single GPU)
    )
"""

from dataclasses import dataclass, field


@dataclass
class MSAConfig:
    """
    Central configuration for the MSA Memory System.

    Sections:
        Core Architecture — Model backbone, attention parameters
        Scale            — Token limits for memory, query, and answer
        Training         — Loss weights, learning rates, curriculum
        Tiered Storage   — GPU/CPU memory allocation (Memory Parallel)
        Memory Interleave — Multi-hop reasoning parameters

    Environment-Specific Defaults:
        GPU (2×A800):  max_memory_tokens=100_000_000 (100M)
        GPU (1×A100):  max_memory_tokens=40_000_000  (40M)
        GPU (1×4090):  max_memory_tokens=10_000_000  (10M)
        CPU only:      max_memory_tokens=10_000_000  (10M)
    """

    # --- Core architecture ---
    backbone_model: str = "Qwen/Qwen3-4B-Instruct"
    """HuggingFace model ID or local path for the backbone transformer."""

    chunk_size: int = 64
    """P: tokens per compression chunk (mean pooling window).
    Smaller = finer-grained retrieval but more memory.
    Larger  = more compression but coarser routing."""

    top_k_docs: int = 16
    """k: number of documents retrieved per query.
    Higher values improve recall at the cost of generation latency."""

    router_layers_fraction: float = 0.5
    """Fraction of model layers (from the end) that apply MSA routing.
    Default 0.5 = latter half only. Early layers lack semantic abstraction
    for effective routing — force-routing them degrades performance."""

    head_dim: int = 128
    """Dimension per attention head. Must match backbone model."""

    num_heads: int = 8
    """Number of attention heads. Must match backbone model."""

    # --- Scale ---
    max_memory_tokens: int = 100_000_000
    """Maximum tokens in the memory bank. 100M = paper's proven ceiling.
    Reduce for smaller GPUs (see Environment-Specific Defaults above)."""

    max_query_tokens: int = 4096
    """Maximum tokens in a single query."""

    max_answer_tokens: int = 1024
    """Maximum tokens in a generated answer."""

    # --- Training ---
    warmup_lm_weight: float = 0.1
    """Phase 1 (warmup): language model loss weight.
    Low weight lets the router learn first."""

    warmup_aux_weight: float = 1.0
    """Phase 1 (warmup): auxiliary contrastive loss weight.
    High weight primes the Router Projectors quickly."""

    main_lm_weight: float = 1.0
    """Phase 2 (main): language model loss weight.
    Generation quality dominates after router is primed."""

    main_aux_weight: float = 0.1
    """Phase 2 (main): auxiliary contrastive loss weight.
    Low weight maintains routing alignment without dominating."""

    warmup_lr: float = 1e-4
    """Learning rate during warmup phase."""

    main_lr: float = 6e-6
    """Learning rate during main training phase."""

    warmup_steps: int = 5000
    """Number of warmup steps. Ablation: removing warmup causes
    31.3% average performance drop."""

    temperature: float = 0.07
    """τ (tau) for contrastive loss (Equation 5 in MSA paper).
    Controls sharpness of the routing distribution."""

    # --- Tiered storage (Memory Parallel) ---
    routing_keys_device: str = "cuda"
    """Device for routing keys (K̄ᴿ). Must be GPU for fast scoring.
    At 100M tokens: ~56 GB VRAM required."""

    content_kvs_device: str = "cpu"
    """Device for content KVs (K̄, V̄). CPU DRAM is sufficient since
    only top-k are fetched per query. At 100M tokens: ~113 GB DRAM."""

    async_prefetch: bool = True
    """Enable async CPU→GPU transfer for selected top-k content KVs.
    Overlaps data transfer with routing computation."""

    # --- Memory Interleave (multi-hop) ---
    max_interleave_rounds: int = 5
    """Maximum retrieve→generate→retrieve cycles for multi-hop reasoning.
    Paper ablation: with interleave = 4.020 on HotpotQA,
    without = 3.250 (−19.2%)."""

    interleave_delimiter: str = "<end_of_retrieve>"
    """Token emitted by the model to signal retrieval completion.
    Must be added to tokenizer's special tokens during training."""
