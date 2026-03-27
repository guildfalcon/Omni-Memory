"""
Large-Scale Example — 100M Token Memory with Tiered Storage.

This example demonstrates Memory Parallel: the tiered GPU/CPU storage
strategy that enables 100M-token scale on 2×A800 GPUs.

Memory budget breakdown:
  - Compressed KV + K̄ᴿ cache: ~169 GB total
  - 2×A800 VRAM:               ~160 GB total
  - Solution: routing keys → GPU, content KVs → CPU DRAM

Key principle: NEVER move all 100M tokens to GPU simultaneously.
Only the top-k selected documents transfer CPU → GPU per query.
"""

from msa_memory.config import MSAConfig
from msa_memory.parallel import TieredMemoryBank


def main():
    print("=" * 60)
    print("Large-Scale Memory Parallel Example")
    print("=" * 60)
    print()

    # =================================================================
    # Configuration for maximum scale
    # =================================================================
    config = MSAConfig(
        backbone_model="Qwen/Qwen3-4B-Instruct",
        max_memory_tokens=100_000_000,  # 100M tokens
        top_k_docs=32,                  # more docs for better recall
        routing_keys_device="cuda",     # K̄ᴿ → GPU VRAM (~56 GB)
        content_kvs_device="cpu",       # K̄, V̄ → CPU DRAM (~113 GB)
        async_prefetch=True,            # overlap transfer with scoring
    )

    print("Scale Configuration:")
    print(f"  Max memory tokens:   {config.max_memory_tokens:,}")
    print(f"  Top-k documents:     {config.top_k_docs}")
    print(f"  Routing keys device: {config.routing_keys_device}")
    print(f"  Content KVs device:  {config.content_kvs_device}")
    print(f"  Async prefetch:      {config.async_prefetch}")
    print()

    # =================================================================
    # Memory budget estimation
    # =================================================================
    tokens = config.max_memory_tokens
    vram_gb = tokens / 100_000_000 * 56
    dram_gb = tokens / 100_000_000 * 113
    total_gb = vram_gb + dram_gb

    print("Estimated Memory Budget:")
    print(f"  Routing keys (VRAM): {vram_gb:.1f} GB")
    print(f"  Content KVs (DRAM):  {dram_gb:.1f} GB")
    print(f"  Total:               {total_gb:.1f} GB")
    print()

    # =================================================================
    # Tiered storage walkthrough
    # =================================================================
    print("Tiered Storage Strategy:")
    print("  1. Load routing keys → GPU VRAM (fast cosine scoring)")
    print("  2. Load content KVs → CPU DRAM (on-demand transfer)")
    print("  3. Per query:")
    print("     a. Score ALL routing keys on GPU → O(M·L/P)")
    print("     b. Select top-k documents")
    print("     c. Transfer ONLY top-k content KVs → GPU")
    print("     d. Generate with sparse context")
    print()

    # =================================================================
    # Multi-GPU distributed scoring
    # =================================================================
    print("Multi-GPU Distributed Scoring:")
    print("  GPU 0: scores documents 0 to N/2")
    print("  GPU 1: scores documents N/2 to N")
    print("  Reduce: merge scores, select global top-k")
    print()
    print("  bank = TieredMemoryBank(config)")
    print("  bank.load('./msa_memory_bank/')")
    print("  scores = bank.distributed_score(query_routing, gpu_ids=[0, 1])")
    print("  top_k = torch.topk(scores, k=32).indices")
    print("  selected_k, selected_v = bank.fetch_top_k(top_k.tolist())")


if __name__ == "__main__":
    main()
