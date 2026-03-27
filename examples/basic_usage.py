"""
Basic Usage Example — Getting started with Omni-Memory.

This example demonstrates:
  1. Loading a pre-encoded memory bank
  2. Searching for relevant documents
  3. Using the stateless memory server

Prerequisites:
  - Run `python -m msa_memory.encode` first to create a memory bank
  - See docs/QUICKSTART.md for full setup instructions
"""

from msa_memory.config import MSAConfig
from msa_memory.parallel import TieredMemoryBank


def main():
    # =================================================================
    # 1. Configure
    # =================================================================
    config = MSAConfig(
        backbone_model="Qwen/Qwen3-4B-Instruct",
        chunk_size=64,
        top_k_docs=16,
        # For CPU-only development:
        routing_keys_device="cpu",
        content_kvs_device="cpu",
    )

    print(f"Configuration:")
    print(f"  Backbone: {config.backbone_model}")
    print(f"  Chunk size: {config.chunk_size}")
    print(f"  Top-k documents: {config.top_k_docs}")
    print()

    # =================================================================
    # 2. Load Memory Bank
    # =================================================================
    print("Loading memory bank...")
    bank = TieredMemoryBank(config)
    bank.load("./msa_memory_bank/")

    print(f"  Documents loaded: {len(bank.kr_cache)}")
    print(f"  Document index entries: {len(bank.doc_index)}")
    print()

    # =================================================================
    # 3. Inspect Document Index
    # =================================================================
    print("Sample documents in memory:")
    for doc_id, text in list(bank.doc_index.items())[:3]:
        preview = text[:100].replace("\n", " ")
        print(f"  [{doc_id}] {preview}...")
    print()

    # =================================================================
    # 4. Use the Stateless Server API
    # =================================================================
    from msa_memory.server import StatelessMemoryServer

    server = StatelessMemoryServer("./msa_memory_bank/", config)
    server.load()

    # Health check (zero DB operations)
    health = server.health_check()
    print(f"Server health: {health}")

    # Query (stateless — no writes)
    results = server.query("What are the user's code preferences?")
    print(f"Query returned {len(results)} results")
    for r in results[:3]:
        print(f"  [{r['id']}] {r.get('content', '')[:80]}...")


if __name__ == "__main__":
    main()
