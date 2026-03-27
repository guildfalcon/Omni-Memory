"""
MSA Memory Integration for Hermes Agent (NousResearch Hermes + tool use).

Hermes uses a structured system prompt + tool-use format.
This plugin injects MSA memory into the Hermes system prompt at session start
and provides tool definitions compatible with Hermes's function calling schema.

Usage:
    from integrations.hermes_plugin import build_hermes_session

    session = build_hermes_session(
        memory_bank=bank,
        initial_query="What are the user's code style preferences?",
        config=config,
    )
"""

from msa_memory.config import MSAConfig
from msa_memory.generate import encode_query_text
from msa_memory.parallel import TieredMemoryBank
from msa_memory.route import route_query


HERMES_SYSTEM_PROMPT_TEMPLATE = """
<|im_start|>system
You are a Hermes AI assistant with MSA persistent memory.

## Memory System
You have access to a latent-space memory bank (MSA — Memory Sparse Attention).
This is NOT a file system or a RAG database. It is a trained sparse attention
mechanism that retrieves from the model's own KV cache representations.

### How to use memory
- Call `msa_search` BEFORE answering any question that might benefit from prior context
- Call `msa_store` AFTER any response containing information worth retaining
- Call `msa_multihop` for questions requiring evidence from multiple documents

### Current session memory stats
Total documents: {total_docs}
Total tokens: {total_tokens}
Memory bank version: {bank_version}
Last updated: {last_updated}

{injected_context}
<|im_end|>
"""

HERMES_TOOL_DEFINITIONS = """
<tools>
[
  {
    "name": "msa_search",
    "description": "Search MSA memory using sparse latent-space routing. Returns top-k relevant document contexts. Always call this before answering questions about prior conversations, user preferences, codebase details, or domain knowledge.",
    "parameters": {
      "type": "object",
      "properties": {
        "query": {"type": "string"},
        "top_k": {"type": "integer", "default": 16},
        "multihop": {"type": "boolean", "default": false, "description": "Enable Memory Interleave for complex multi-step questions"}
      },
      "required": ["query"]
    }
  },
  {
    "name": "msa_store",
    "description": "Store new information in MSA memory bank for future retrieval.",
    "parameters": {
      "type": "object",
      "properties": {
        "content": {"type": "string"},
        "category": {"type": "string", "enum": ["preference", "fact", "decision", "code", "conversation"]},
        "priority": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.5}
      },
      "required": ["content"]
    }
  },
  {
    "name": "msa_multihop",
    "description": "Run Memory Interleave for multi-hop reasoning. Use for questions like 'what is the relationship between X and Y' or 'who worked on the project that used technology Z'.",
    "parameters": {
      "type": "object",
      "properties": {
        "query": {"type": "string"},
        "max_rounds": {"type": "integer", "default": 5}
      },
      "required": ["query"]
    }
  }
]
</tools>
"""


def build_hermes_session(
    memory_bank: TieredMemoryBank,
    initial_query: str,
    config: MSAConfig,
) -> str:
    """
    Build the complete Hermes session opening with MSA context injected.

    Pre-fetches top-k context for the initial query to warm the session,
    then formats the full system prompt with tool definitions.

    Args:
        memory_bank: Loaded TieredMemoryBank instance.
        initial_query: The first user query to pre-fetch context for.
        config: MSAConfig instance.

    Returns:
        Formatted Hermes system prompt string with tools and injected context.
    """
    # Pre-fetch top-k context for initial query to warm the session
    if initial_query:
        query_hidden = encode_query_text(
            initial_query, memory_bank.model, memory_bank.tokenizer
        )
        K_ctx, V_ctx, top_k_ids = route_query(
            query_hidden,
            memory_bank.kr_cache,
            memory_bank.k_cache,
            memory_bank.v_cache,
            config,
            memory_bank.router_q_proj,
        )
        retrieved = [
            memory_bank.doc_index.get(str(i), "")[:800] for i in top_k_ids
        ]
        injected = "\n\n".join(
            f"[{i+1}] {t}" for i, t in enumerate(retrieved) if t
        )
    else:
        injected = ""

    return HERMES_SYSTEM_PROMPT_TEMPLATE.format(
        total_docs=len(memory_bank.kr_cache),
        total_tokens=len(memory_bank.kr_cache) * 512,  # estimated
        bank_version="1.0",
        last_updated="session start",
        injected_context=(
            f"## Pre-retrieved context\n{injected}" if injected else ""
        ),
    ) + HERMES_TOOL_DEFINITIONS
