"""
Stage 2 — Online Routing & Context Assembly.

Given a user query, this module identifies the top-k most relevant documents
from the encoded corpus and assembles them into a sparse context for generation.

The routing process:
  1. Project query hidden states to routing space (QR)
  2. Compute cosine similarity between QR and each document's routing key (K̄ᴿ)
  3. Select top-k documents by relevance score
  4. Assemble sparse context: [{K̄ᵢ}ᵢ∈I; Kq]

Cost: O(M·L/P) — linear in corpus size L, compressed by chunk factor P.

Key implementation detail (from paper):
  - Sᵢⱼ = max_token_t( mean_head_h( cos(QᴿQ,ₕ)ₜ, K̄ᴿᵢⱼ,ₕ) ) )
  - sᵢ  = max_j(Sᵢⱼ)   → document-level score
  - I   = Top-k({sᵢ})   → selected documents
"""

from typing import Tuple

import torch
import torch.nn.functional as F

from .config import MSAConfig


def route_query(
    query_hidden: torch.Tensor,          # [1, q_len, d_model]
    kr_cache: list[torch.Tensor],        # List of [1, n_chunks, d] per document
    k_cache: list[torch.Tensor],         # List of [1, n_chunks, d] per document
    v_cache: list[torch.Tensor],         # List of [1, n_chunks, d] per document
    config: MSAConfig,
    router_q_proj,                       # nn.Linear: d_model → d_routing
) -> Tuple[torch.Tensor, torch.Tensor, list[int]]:
    """
    Stage 2: Given query hidden states, identify top-k documents
    and assemble the sparse context for generation.

    Args:
        query_hidden: Query hidden states from the backbone model [1, q_len, d_model].
        kr_cache: List of compressed routing keys per document.
        k_cache: List of compressed standard keys per document.
        v_cache: List of compressed standard values per document.
        config: MSAConfig instance.
        router_q_proj: Projection layer from query space to routing space.

    Returns:
        Tuple of (K_ctx, V_ctx, selected_doc_ids) where:
          - K_ctx: assembled sparse key context [1, total_chunks + q_len, d]
          - V_ctx: assembled sparse value context [1, total_chunks + q_len, d]
          - selected_doc_ids: list of selected document indices
    """
    # Project query to routing space
    QR = router_q_proj(query_hidden)  # [1, q_len, d_routing]
    QR = F.normalize(QR, dim=-1)

    doc_scores = []
    for doc_id, KR_doc in enumerate(kr_cache):
        # Move routing key to same device as query for scoring
        KR_doc = KR_doc.to(query_hidden.device)
        KR_doc = F.normalize(KR_doc, dim=-1)

        # Cosine similarity: [1, q_len, n_chunks]
        sim = torch.einsum("bqd,bcd->bqc", QR, KR_doc)

        # Mean over heads, max over query tokens and chunks
        # Equation (2) from paper: Sᵢⱼ = max_t(mean_h(cos(QᴿQ,t, K̄ᴿᵢⱼ)))
        chunk_score = sim.mean(0).max(0).values  # [n_chunks]
        doc_score = chunk_score.max().item()     # scalar — document-level score
        doc_scores.append(doc_score)

    # Select top-k document indices
    scores_tensor = torch.tensor(doc_scores)
    top_k_indices = torch.topk(
        scores_tensor, k=min(config.top_k_docs, len(doc_scores))
    ).indices.tolist()

    # Assemble sparse context: [{K̄ᵢ}ᵢ∈I; Kq]
    selected_K = [k_cache[i] for i in top_k_indices]
    selected_V = [v_cache[i] for i in top_k_indices]

    K_ctx = torch.cat(
        selected_K + [query_hidden.to(query_hidden.device)], dim=1
    )
    V_ctx = torch.cat(
        selected_V + [query_hidden.to(query_hidden.device)], dim=1
    )

    return K_ctx, V_ctx, top_k_indices
