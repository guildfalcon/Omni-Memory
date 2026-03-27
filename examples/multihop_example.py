"""
Multi-Hop Reasoning Example — Using Memory Interleave.

This example demonstrates multi-hop reasoning where single-shot retrieval
fails but iterative Memory Interleave succeeds.

Scenario:
  Query: "Who is the son of the founder of X?"

  Single-shot RAG:
    → Retrieves docs about X, finds "X was founded by Y"
    → Cannot find Y's son because it only searched once
    → Wrong answer

  Memory Interleave:
    → Round 1: Retrieves docs about X → "X was founded by Y"
    → Round 2: Retrieves docs about Y → "Y's son is Z, born 1985"
    → Correct answer: Z, born 1985

Prerequisites:
  - Run `python -m msa_memory.encode` first to create a memory bank
  - Memory bank should contain documents about X, Y, and Z
"""

from msa_memory.config import MSAConfig


def main():
    config = MSAConfig(
        backbone_model="Qwen/Qwen3-4B-Instruct",
        max_interleave_rounds=5,
        top_k_docs=16,
    )

    print("=" * 60)
    print("Multi-Hop Memory Interleave Example")
    print("=" * 60)
    print()

    # Example query requiring multi-hop reasoning
    query = "Who is the son of the founder of Acme Corp?"
    print(f"Query: {query}")
    print()

    # Demonstrate the interleave logic
    print("Memory Interleave Rounds:")
    print()

    # Round 1: Find information about Acme Corp
    print("  Round 1:")
    print("    Search: 'Acme Corp founder'")
    print("    Found:  [Doc #42] 'Acme Corp was founded by John Smith in 1995'")
    print("    → Need more info about John Smith")
    print()

    # Round 2: Find information about John Smith
    print("  Round 2:")
    print("    Search: 'John Smith family'")
    print("    Found:  [Doc #87] 'John Smith has a son named Alex Smith, born 1985'")
    print("    → Model emits <end_of_retrieve>")
    print()

    # Final answer
    print("  Final Answer:")
    print("    'Alex Smith, born 1985, is the son of John Smith,")
    print("     who founded Acme Corp in 1995.'")
    print()

    print("-" * 60)
    print()

    # Show the config options for multi-hop
    print("Configuration for multi-hop:")
    print(f"  max_interleave_rounds: {config.max_interleave_rounds}")
    print(f"  interleave_delimiter:  {config.interleave_delimiter}")
    print(f"  top_k_docs:            {config.top_k_docs}")
    print()

    print("To run with your own data:")
    print("  from msa_memory.interleave import memory_interleave")
    print("  answer = memory_interleave(")
    print("      initial_query='your complex question',")
    print("      doc_index=bank.doc_index,")
    print("      kr_cache=bank.kr_cache,")
    print("      k_cache=bank.k_cache,")
    print("      v_cache=bank.v_cache,")
    print("      model=model,")
    print("      tokenizer=tokenizer,")
    print("      router_q_proj=router_q_proj,")
    print("      config=config,")
    print("  )")


if __name__ == "__main__":
    main()
