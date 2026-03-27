"""
Patch: Precision Ceiling (RAG Semantic Gap).

Supermemory failure: model-agnostic embeddings impose a precision ceiling.
Fix: replace external embedding lookup with in-model latent routing.

Supermemory (broken):
  query_embedding = external_embedder.encode(query)   # model-agnostic
  results = vector_db.search(query_embedding)         # no joint optimisation

MSA (fixed):
  query_hidden = model.forward(query)                 # model's own representations
  top_k = route_query(query_hidden, kr_cache)         # latent-space routing
  # Retrieval and generation share the same optimisation objective
"""

from msa_memory.config import MSAConfig
from msa_memory.generate import encode_query_text
from msa_memory.parallel import TieredMemoryBank
from msa_memory.route import route_query


class PrecisionPatch:
    """
    Drop-in replacement for Supermemory's search() endpoint.
    Uses MSA latent routing instead of external vector similarity.

    Usage:
        patch = PrecisionPatch(memory_bank, config)
        results = patch.search("What are the user's code preferences?")
        # Returns same format as supermemory_client.search()
    """

    def __init__(self, memory_bank: TieredMemoryBank, config: MSAConfig):
        self.memory_bank = memory_bank
        self.config = config

    def search(self, query: str, top_k: int = 16) -> list[dict]:
        """
        Replaces: supermemory_client.search(query)
        Returns same format for drop-in compatibility.

        Args:
            query: Natural language search query.
            top_k: Number of documents to retrieve.

        Returns:
            List of dicts with 'id', 'content', and 'score' keys.
        """
        query_hidden = encode_query_text(
            query,
            self.memory_bank.model,
            self.memory_bank.tokenizer,
        )
        K_ctx, V_ctx, selected_ids = route_query(
            query_hidden,
            self.memory_bank.kr_cache,
            self.memory_bank.k_cache,
            self.memory_bank.v_cache,
            self.config,
            self.memory_bank.router_q_proj,
        )
        return [
            {
                "id": str(i),
                "content": self.memory_bank.doc_index.get(str(i), ""),
                "score": (
                    float(self.memory_bank.scores[i])
                    if hasattr(self.memory_bank, "scores")
                    else 1.0
                ),
            }
            for i in selected_ids
        ]
