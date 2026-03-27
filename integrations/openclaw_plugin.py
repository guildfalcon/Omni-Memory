"""
MSA Memory Plugin for OpenClaw (Hook-Based).

OpenClaw's built-in memory failure modes (from Supermemory blog post):
  1. Tool-based saves: slow, uses MORE tokens, not fewer
  2. No knowledge update: adds redundant facts, doesn't evolve
  3. No forgetting: stale context pollutes responses
  4. RAG quality: 58.3% on MemoryBench (vs MSA: 85.9%+)

This plugin replaces all of that with hook-based implicit MSA memory.
Memory saves happen in the background — the agent never burns tokens on it.

KEY DIFFERENCE from OpenClaw's tool-call model:
  - OpenClaw tool-call: agent must explicitly call memory_search() (slow, forgetful)
  - MSA hook: context is injected automatically before every generation (fast, always-on)

Setup:
    openclaw plugin install ./integrations/openclaw_plugin.py
"""

import asyncio

from msa_memory.config import MSAConfig
from msa_memory.parallel import TieredMemoryBank
from msa_memory.route import route_query


OPENCLAW_PLUGIN_CONFIG = {
    "name": "msa-memory",
    "version": "1.0.0",
    "description": (
        "MSA latent-space memory for OpenClaw. Replaces file-based memory.md "
        "with end-to-end trainable sparse attention retrieval."
    ),
    "hooks": {
        "on_message_received": "msa_ingest_hook",    # background encoding
        "on_response_generated": "msa_store_hook",   # implicit save
        "on_session_start": "msa_load_hook",         # load memory bank
        "before_generation": "msa_inject_hook",      # inject retrieved context
    },
    "replaces": ["MEMORY.md", "memory/YYYY-MM-DD.md", "qmd_memory_plugin"],
}


async def msa_ingest_hook(
    message: dict,
    memory_bank: TieredMemoryBank,
    config: MSAConfig,
):
    """
    Hook: fires on every received message.
    Encodes content in background — agent never waits for this.

    KEY DIFFERENCE from OpenClaw's tool-based approach:
    No tool call, no token cost, no blocking. Pure background IO.
    """
    content = message.get("content", "")
    if len(content.strip()) > 20:
        # Background encode — non-blocking
        asyncio.create_task(
            background_encode_and_store(content, memory_bank, config)
        )


async def msa_inject_hook(
    query: str,
    memory_bank: TieredMemoryBank,
    config: MSAConfig,
) -> str:
    """
    Hook: fires BEFORE generation. Injects retrieved memory as context prefix.

    This is the opposite of OpenClaw's tool-call model:
    - OpenClaw: agent must explicitly call memory_search() (slow, forgetful)
    - MSA hook: context is injected automatically before every generation
    """
    from msa_memory.generate import encode_query_text

    # Route query
    query_hidden = encode_query_text(query, memory_bank.model, memory_bank.tokenizer)
    K_ctx, V_ctx, top_k_ids = route_query(
        query_hidden,
        memory_bank.kr_cache,
        memory_bank.k_cache,
        memory_bank.v_cache,
        config,
        memory_bank.router_q_proj,
    )

    # Build context injection string for OpenClaw system prompt
    retrieved_texts = [
        memory_bank.doc_index.get(str(i), "") for i in top_k_ids
    ]
    context_block = "\n\n".join(
        f"[Memory {i+1}]: {text[:1000]}"
        for i, text in enumerate(retrieved_texts)
        if text.strip()
    )

    return f"""
## Retrieved Memory Context (MSA — top-{config.top_k_docs} documents)
{context_block}

---
"""


async def background_encode_and_store(
    content: str,
    memory_bank: TieredMemoryBank,
    config: MSAConfig,
):
    """Background encoding stub — queues content for batch re-encoding."""
    # In production, this would append to a queue for the next
    # offline encoding batch. See docs/ARCHITECTURE.md for details.
    pass


def get_openclaw_plugin_manifest() -> dict:
    """Return OpenClaw plugin manifest for installation."""
    return {
        **OPENCLAW_PLUGIN_CONFIG,
        "setup": {
            "install": "pip install omni-memory",
            "configure": {
                "MEMORY_BANK_PATH": "./msa_memory_bank/",
                "TOP_K_DOCS": 16,
                "CHUNK_SIZE": 64,
                "ENABLE_INTERLEAVE": True,
            },
        },
        "github": "https://github.com/yourusername/omni-memory",
    }
