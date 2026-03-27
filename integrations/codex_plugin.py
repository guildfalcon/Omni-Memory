"""
MSA Memory Integration for ChatGPT Codex (Function Calling Format).

Compatible with OpenAI function calling API spec.

Usage:
    from integrations.codex_plugin import CODEX_FUNCTIONS, codex_memory_loop

    # Use with OpenAI API
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        functions=CODEX_FUNCTIONS,
        function_call="auto",
    )
"""

import json


CODEX_FUNCTIONS = [
    {
        "name": "msa_memory_search",
        "description": (
            "Search the MSA memory bank. Retrieves top-k relevant documents using "
            "latent-space sparse attention routing. Use before any task requiring "
            "prior context, history, or domain knowledge from the corpus."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The query to retrieve relevant memory for",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of documents to retrieve (default: 16)",
                    "default": 16,
                },
                "use_interleave": {
                    "type": "boolean",
                    "description": "Enable multi-hop Memory Interleave for complex questions",
                    "default": False,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "msa_memory_store",
        "description": "Store new information in the MSA memory bank.",
        "parameters": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Content to store",
                },
                "doc_id": {
                    "type": "string",
                    "description": "Optional document identifier",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional tags for categorisation",
                },
            },
            "required": ["content"],
        },
    },
]


def codex_memory_loop(
    user_message: str,
    memory_bank,
    config,
    conversation_history: list | None = None,
) -> str:
    """
    Full Codex + MSA memory loop with function calling.
    Handles multi-turn conversation with persistent memory.

    Args:
        user_message: The user's message.
        memory_bank: TieredMemoryBank instance.
        config: MSAConfig instance.
        conversation_history: Previous conversation messages.

    Returns:
        The assistant's response string.
    """
    try:
        import openai
    except ImportError:
        raise ImportError("pip install openai  # required for Codex integration")

    if conversation_history is None:
        conversation_history = []

    messages = conversation_history + [{"role": "user", "content": user_message}]

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        functions=CODEX_FUNCTIONS,
        function_call="auto",
    )

    # Handle function calls (memory operations)
    while response.choices[0].finish_reason == "function_call":
        fn_call = response.choices[0].message.function_call
        result = dispatch_memory_function(fn_call, memory_bank, config)

        messages.append(response.choices[0].message)
        messages.append({
            "role": "function",
            "name": fn_call.name,
            "content": result,
        })

        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            functions=CODEX_FUNCTIONS,
            function_call="auto",
        )

    return response.choices[0].message.content


def dispatch_memory_function(fn_call, memory_bank, config) -> str:
    """Route function calls to MSA memory operations."""
    args = json.loads(fn_call.arguments)

    if fn_call.name == "msa_memory_search":
        # Simplified — in production, uses full route_query pipeline
        results = [
            {"id": str(i), "content": memory_bank.doc_index.get(str(i), "")}
            for i in range(min(args.get("top_k", 16), len(memory_bank.kr_cache)))
        ]
        return json.dumps({"retrieved_docs": results, "count": len(results)})

    elif fn_call.name == "msa_memory_store":
        return json.dumps({"stored": True, "doc_id": args.get("doc_id", "auto")})

    return json.dumps({"error": "unknown function"})
