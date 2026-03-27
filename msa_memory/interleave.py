"""
Memory Interleave — Multi-Hop Reasoning.

This is MSA's answer to the multi-hop problem that single-shot RAG cannot solve.

Core idea: Instead of one retrieval pass, we:
  1. Generate document IDs relevant to the query
  2. Load those documents and append to context
  3. Re-query with enriched context
  4. Repeat until model emits <end_of_retrieve>
  5. Then generate the final answer

This enables chained evidence reasoning:
  Query: "Who is the son of the founder of X?"
  Round 1: retrieve docs about X → finds "X was founded by Y"
  Round 2: retrieve docs about Y → finds "Y's son is Z, born 1985"
  Answer: Z, born 1985

Ablation study result from paper:
  - With Memory Interleave:    HotpotQA = 4.020
  - Without Memory Interleave: HotpotQA = 3.250  (−19.2%)
"""

import re

from .config import MSAConfig
from .generate import encode_query_text, generate_sparse
from .route import route_query


def memory_interleave(
    initial_query: str,
    doc_index: dict[str, str],   # id → text loaded from doc_index.json
    kr_cache,
    k_cache,
    v_cache,
    model,
    tokenizer,
    router_q_proj,
    config: MSAConfig,
    system_prompt: str = "",
) -> str:
    """
    Run the full Memory Interleave loop.

    This function implements iterative retrieve → generate → retrieve cycles
    for complex multi-hop reasoning. The model decides when to stop retrieving
    by emitting the interleave_delimiter token.

    Args:
        initial_query: The user's natural language question.
        doc_index: Mapping of document ID → document text (from doc_index.json).
        kr_cache: List of compressed routing keys per document.
        k_cache: List of compressed standard keys per document.
        v_cache: List of compressed standard values per document.
        model: The backbone language model.
        tokenizer: Corresponding tokenizer.
        router_q_proj: Router query projection layer.
        config: MSAConfig instance.
        system_prompt: Optional system prompt prefix.

    Returns:
        The final answer string after multi-hop reasoning.
    """
    context_docs = []        # accumulates retrieved document texts
    rounds = 0

    while rounds < config.max_interleave_rounds:
        rounds += 1

        # Build enriched prompt: original query + all retrieved docs so far
        enriched_prompt = build_interleave_prompt(
            query=initial_query,
            retrieved_docs=context_docs,
            system_prompt=system_prompt,
            delimiter=config.interleave_delimiter,
        )

        # Route query with enriched context
        query_hidden = encode_query_text(enriched_prompt, model, tokenizer)
        K_ctx, V_ctx, selected_ids = route_query(
            query_hidden, kr_cache, k_cache, v_cache, config, router_q_proj
        )

        # Generate: either doc IDs (still retrieving) or final answer
        generated = generate_sparse(
            query_hidden, K_ctx, V_ctx, model, tokenizer, config
        )

        # Check if model signals retrieval is complete
        if config.interleave_delimiter in generated:
            # Extract document IDs from generated text
            doc_ids = extract_doc_ids(generated)

            if not doc_ids:
                # No more docs to retrieve — generate final answer
                break

            # Load document texts and add to context
            for doc_id in doc_ids:
                if str(doc_id) in doc_index:
                    context_docs.append({
                        "id": doc_id,
                        "text": doc_index[str(doc_id)],
                    })
        else:
            # Model generated the final answer directly
            return clean_answer(generated)

    # Final answer generation pass with full accumulated context
    final_prompt = build_final_prompt(initial_query, context_docs, system_prompt)
    final_hidden = encode_query_text(final_prompt, model, tokenizer)
    K_ctx, V_ctx, _ = route_query(
        final_hidden, kr_cache, k_cache, v_cache, config, router_q_proj
    )
    return generate_sparse(final_hidden, K_ctx, V_ctx, model, tokenizer, config)


def build_interleave_prompt(
    query: str,
    retrieved_docs: list,
    system_prompt: str,
    delimiter: str,
) -> str:
    """Format the prompt for an interleave round."""
    doc_context = ""
    for doc in retrieved_docs:
        doc_context += f"\n[{doc['id']}] {doc['text']}\n"

    return f"""{system_prompt}

Retrieved context so far:
{doc_context}

Query: {query}

Generate relevant document IDs to retrieve next, or generate the final answer \
if you have enough context.
End retrieval with: {delimiter}
"""


def build_final_prompt(
    query: str,
    context_docs: list,
    system_prompt: str,
) -> str:
    """Build the final generation prompt with all accumulated context."""
    doc_context = ""
    for doc in context_docs:
        doc_context += f"\n[{doc['id']}] {doc['text']}\n"

    return f"""{system_prompt}

All retrieved context:
{doc_context}

Query: {query}

Provide a comprehensive answer based on the retrieved context above.
"""


def extract_doc_ids(generated_text: str) -> list[int]:
    """Parse document IDs from model output like: [4] [7] [12] <end_of_retrieve>"""
    ids = re.findall(r"\[(\d+)\]", generated_text)
    return [int(i) for i in ids]


def clean_answer(text: str) -> str:
    """Remove any retrieval artifacts from generated answer text."""
    # Remove delimiter tokens
    text = text.replace("<end_of_retrieve>", "")
    # Remove doc ID references
    text = re.sub(r"\[\d+\]", "", text)
    return text.strip()
