"""
Doc-wise RoPE Utilities — The Positional Generalisation Fix.

This module implements the positional encoding strategy that allows MSA to
train on 64K context and extrapolate to 100M tokens at inference.

The problem with global RoPE:
  Standard positional encoding assigns IDs 0, 1, 2 … N*doc_len across all
  concatenated documents. When inference N >> training N, position indices
  explode → catastrophic degradation.

The MSA fix:
  - Doc-wise RoPE: Each document's position IDs reset to 0 independently
  - Query RoPE: Global offset by k (number of retrieved docs) so the model
    perceives the query as a logical continuation of retrieved context

This is the core mechanism that enables train-on-64K → infer-on-100M.
"""

import torch


def build_docwise_position_ids(
    doc_lengths: list[int],
) -> torch.Tensor:
    """
    Build document-wise independent positional IDs.

    Each document's positions reset to 0, preventing position index explosion
    when the number of documents grows at inference time.

    Args:
        doc_lengths: List of token counts per document.

    Returns:
        Position IDs tensor [1, total_tokens] with per-document resets.

    Example:
        >>> build_docwise_position_ids([100, 200, 150])
        # Returns: [0,1,...,99, 0,1,...,199, 0,1,...,149]
        # NOT:     [0,1,...,99, 100,...,299, 300,...,449]  (global — WRONG)
    """
    position_ids = []
    for doc_len in doc_lengths:
        position_ids.append(torch.arange(doc_len))
    return torch.cat(position_ids).unsqueeze(0)  # [1, total_tokens]


def build_query_position_ids(
    query_len: int,
    num_retrieved_docs: int,
) -> torch.Tensor:
    """
    Build global RoPE position IDs for the active query context.

    The query is offset by k (number of top-k retrieved docs) so the model
    perceives it as a logical continuation of the retrieved context.

    Args:
        query_len: Number of tokens in the query.
        num_retrieved_docs: Number of retrieved documents (k).

    Returns:
        Position IDs tensor [1, query_len] starting at offset k.

    Example:
        >>> build_query_position_ids(query_len=512, num_retrieved_docs=16)
        # Returns: [16, 17, 18, ..., 527]
    """
    return torch.arange(
        num_retrieved_docs,
        num_retrieved_docs + query_len,
    ).unsqueeze(0)  # [1, query_len]


def build_combined_position_ids(
    doc_lengths: list[int],
    query_len: int,
) -> torch.Tensor:
    """
    Build combined position IDs for the full sparse context.

    Concatenates doc-wise position IDs (independent per document)
    with query position IDs (globally offset by number of docs).

    Args:
        doc_lengths: List of token counts per retrieved document.
        query_len: Number of tokens in the query.

    Returns:
        Combined position IDs [1, sum(doc_lengths) + query_len].
    """
    doc_positions = build_docwise_position_ids(doc_lengths)
    query_positions = build_query_position_ids(query_len, len(doc_lengths))
    return torch.cat([doc_positions, query_positions], dim=1)


def apply_docwise_rope(
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
    rope_fn,
) -> torch.Tensor:
    """
    Apply RoPE with document-wise position IDs.

    This is a thin wrapper that ensures the correct position IDs are used
    when calling the model's native RoPE implementation.

    Args:
        hidden_states: Hidden states to apply positional encoding to.
        position_ids: Doc-wise or query position IDs.
        rope_fn: The model's native RoPE function (varies by backbone).

    Returns:
        Position-encoded hidden states.
    """
    return rope_fn(hidden_states, position_ids)
