"""
MSA Memory Plugin for Claude Code.

Drop this into your CLAUDE.md or pass as --context to get persistent memory.

Fixes the core Claude Code memory failure:
  - Filesystem (CLAUDE.md): 54.2% accuracy on MemoryBench
  - MSA (this plugin):      85.9%+ with training, higher at inference scale

Setup:
  1. Add the system prompt to your CLAUDE.md
  2. Register the tools with Claude Code's tool_use format
  3. The MSA memory bank handles all storage and retrieval

Usage:
    from integrations.claude_code_plugin import (
        CLAUDE_CODE_SYSTEM_PROMPT,
        register_claude_code_tools,
    )

    # Add to CLAUDE.md system prompt
    print(CLAUDE_CODE_SYSTEM_PROMPT)

    # Register tools
    tools = register_claude_code_tools()
"""

CLAUDE_CODE_SYSTEM_PROMPT = """
You have access to a persistent MSA memory system. Before answering any question
about prior work, decisions, codebase preferences, or user history:

1. Call: memory_search(query="<your question>")
2. The system returns top-k document KVs assembled from your corpus
3. Use these as grounding context for your response

Memory is stored in latent KV-cache format (NOT plain text files).
This means:
  - Retrieval is semantically aligned with your reasoning (no precision gap)
  - Knowledge updates are handled via corpus re-encoding (offline)
  - Multi-hop evidence chaining is supported via Memory Interleave
  - You should NEVER fall back to CLAUDE.md markdown files for memory

To add new memory: memory_store(text="<content>", doc_id="<unique_id>")
To run multi-hop retrieval: memory_interleave(query="<complex question>")
"""


def register_claude_code_tools() -> list[dict]:
    """Return tool definitions for Claude Code's tool_use format."""
    return [
        {
            "name": "memory_search",
            "description": (
                "Search MSA memory bank using latent-space routing. "
                "Use for any question requiring prior context, user preferences, "
                "or historical decisions."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language query",
                    },
                    "top_k": {
                        "type": "integer",
                        "default": 16,
                        "description": "Number of documents to retrieve",
                    },
                },
                "required": ["query"],
            },
        },
        {
            "name": "memory_store",
            "description": (
                "Add new document to the MSA memory bank "
                "(triggers re-encoding)."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Content to store in memory",
                    },
                    "doc_id": {
                        "type": "string",
                        "description": "Unique document identifier",
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Optional metadata tags",
                    },
                },
                "required": ["text"],
            },
        },
        {
            "name": "memory_interleave",
            "description": (
                "Run multi-hop Memory Interleave for complex reasoning "
                "over distributed evidence."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Complex question requiring multi-hop reasoning",
                    },
                    "max_rounds": {
                        "type": "integer",
                        "default": 5,
                        "description": "Maximum retrieval rounds",
                    },
                },
                "required": ["query"],
            },
        },
    ]
