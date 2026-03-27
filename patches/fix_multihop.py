"""
Patch: Multi-Hop Failure.

Supermemory failure: single-shot retrieval cannot chain evidence.
Fix: Memory Interleave (Section 3.5 of MSA paper).

Example of the failure:
  Supermemory (broken):
    results = search("who is the son of the founder of X?")
    # Returns documents about X's founder, but misses the son's doc
    # No iteration → wrong answer

  MSA (fixed):
    # Round 1: retrieve docs about X → finds "X was founded by Y"
    # Round 2: retrieve docs about Y → finds "Y's son is Z, born 1985"
    # Answer: Z, born 1985
"""

from msa_memory.config import MSAConfig
from msa_memory.interleave import memory_interleave
from msa_memory.parallel import TieredMemoryBank


def multihop_search(
    query: str,
    memory_bank: TieredMemoryBank,
    config: MSAConfig,
    max_rounds: int = 5,
) -> str:
    """
    Replaces: supermemory_client.search(complex_query) [which fails at multi-hop]

    Uses Memory Interleave for iterative retrieve → generate → retrieve cycles,
    enabling complex multi-document reasoning that single-shot RAG cannot do.

    Args:
        query: Complex question requiring multi-hop reasoning.
        memory_bank: Loaded TieredMemoryBank instance.
        config: MSAConfig instance.
        max_rounds: Maximum retrieval rounds.

    Returns:
        Final answer string after multi-hop reasoning.
    """
    return memory_interleave(
        initial_query=query,
        doc_index=memory_bank.doc_index,
        kr_cache=memory_bank.kr_cache,
        k_cache=memory_bank.k_cache,
        v_cache=memory_bank.v_cache,
        model=memory_bank.model,
        tokenizer=memory_bank.tokenizer,
        router_q_proj=memory_bank.router_q_proj,
        config=config,
    )
