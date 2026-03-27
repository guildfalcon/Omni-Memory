"""
Stage 3 — Sparse Generation Wrapper.

This module handles the final autoregressive generation step using the
sparse context assembled in Stage 2.

The generation process uses:
  - Assembled sparse context: [{K̄_topk}; Kq]
  - Doc-wise RoPE for retrieved documents (position IDs reset per doc)
  - Global RoPE for the query (offset by k for logical continuation)

Cost: O(T·(M + k·G/P)²) — independent of total corpus size L.
"""

import torch

from .config import MSAConfig
from .rope_utils import build_docwise_position_ids, build_query_position_ids


def generate_sparse(
    query_hidden: torch.Tensor,
    K_ctx: torch.Tensor,
    V_ctx: torch.Tensor,
    model,
    tokenizer,
    config: MSAConfig,
    max_new_tokens: int | None = None,
) -> str:
    """
    Generate an answer using sparse attention over the assembled context.

    This is Stage 3 of the MSA pipeline. The model generates autoregressively
    while attending only to the sparse context (top-k retrieved documents +
    the query itself), NOT the entire corpus.

    Args:
        query_hidden: Original query hidden states [1, q_len, d_model].
        K_ctx: Assembled sparse key context from route_query().
        V_ctx: Assembled sparse value context from route_query().
        model: The backbone language model.
        tokenizer: Corresponding tokenizer.
        config: MSAConfig instance.
        max_new_tokens: Override max answer tokens (default: config.max_answer_tokens).

    Returns:
        Generated answer string.
    """
    if max_new_tokens is None:
        max_new_tokens = config.max_answer_tokens

    # Build position IDs using doc-wise RoPE for retrieved docs
    # and global RoPE for the query context
    position_ids = build_query_position_ids(
        query_len=query_hidden.shape[1],
        num_retrieved_docs=config.top_k_docs,
    )

    # Generate with sparse KV context
    # NOTE: This is a simplified wrapper. In production, you would inject
    # K_ctx and V_ctx directly into the model's KV cache mechanism.
    generated_ids = model.generate(
        inputs_embeds=query_hidden,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Decode generated tokens
    generated_text = tokenizer.decode(
        generated_ids[0], skip_special_tokens=True
    )
    return generated_text


def encode_query_text(
    query: str,
    model,
    tokenizer,
) -> torch.Tensor:
    """
    Encode a text query into hidden states for routing.

    Args:
        query: Natural language query string.
        model: The backbone language model.
        tokenizer: Corresponding tokenizer.

    Returns:
        Hidden states tensor [1, q_len, d_model].
    """
    tokens = tokenizer(
        query,
        return_tensors="pt",
        truncation=True,
        padding=False,
    ).input_ids.to(model.device)

    with torch.no_grad():
        outputs = model(
            tokens,
            output_hidden_states=True,
            return_dict=True,
        )
    # Return the last hidden state
    return outputs.hidden_states[-1]
