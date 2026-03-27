# Architecture Deep-Dive

This document explains the full MSA (Memory Sparse Attention) architecture that powers Omni-Memory.

> Built from: *MSA: Memory Sparse Attention for Efficient End-to-End Memory Model Scaling to 100M Tokens* (Evermind / Shanda Group / Peking University)

---

## Overview

MSA separates memory into three cleanly decoupled stages. Each stage has a well-defined input, output, and cost profile:

```
┌─────────────────────────────────────────────────────────────┐
│  STAGE 1 — Global Memory Encoding  (offline, run once)      │
│  Input : Raw document corpus                                 │
│  Output: Compressed KV cache + routing keys (K̄, V̄, K̄ᴿ)    │
│  Cost  : O(L·G)  amortised across all future queries        │
└───────────────────────┬─────────────────────────────────────┘
                        │  cached to disk / DRAM
┌───────────────────────▼─────────────────────────────────────┐
│  STAGE 2 — Routing & Context Assembly  (online, per query)  │
│  Input : User query hidden state Hq                         │
│  Output: Top-k document KVs assembled as sparse context     │
│  Cost  : O(M·L/P)  linear in corpus size L                  │
└───────────────────────┬─────────────────────────────────────┘
                        │  sparse context
┌───────────────────────▼─────────────────────────────────────┐
│  STAGE 3 — Sparse Generation  (online, autoregressive)      │
│  Input : Sparse context [{K̄_topk}; Kq]                     │
│  Output: Final answer tokens                                 │
│  Cost  : O(T·(M + k·G/P)²)  independent of L               │
└─────────────────────────────────────────────────────────────┘
```

---

## The Sparse Attention Mechanism (Core Math)

For each document `dᵢ` with hidden state `Hᵢ`, three projections are computed:

```
Kᵢ,ₕ  = Hᵢ · WᴷH       # standard key
Vᵢ,ₕ  = Hᵢ · WᵛH       # standard value
KᴿI,ₕ = Hᵢ · WᴷᴿH      # routing key (NEW — not in standard transformers)
```

### Chunk-wise Mean Pooling

Compresses each projection into fixed-size latent vectors:

```
K̄ᵢ,ₕ  = φ(Kᵢ,ₕ)      # φ = chunk_mean_pool(·, chunk_size)
V̄ᵢ,ₕ  = φ(Vᵢ,ₕ)
K̄ᴿᵢ,ₕ = φ(KᴿI,ₕ)
```

### Relevance Scoring

At query time, relevance is scored per-document:

```
Sᵢⱼ = max_token_t ( mean_head_h ( cos(QᴿQ,ₕ)ₜ, K̄ᴿᵢⱼ,ₕ) ) )
sᵢ  = max_j(Sᵢⱼ)   # document-level score
I   = Top-k({sᵢ})   # select top-k documents
```

### Final Sparse Attention

```
Kctx = [{K̄ᵢ}ᵢ∈I ; Kq]
Vctx = [{V̄ᵢ}ᵢ∈I ; Vq]
Output = Attention(Qq, Kctx, Vctx)
```

---

## Critical Implementation Notes

### MSA Layer Placement

> **Apply MSA routing ONLY to the latter half of the model's layers.**

Early layers lack the semantic abstraction needed for effective routing — force-routing them degrades performance. The `router_layers_fraction` config parameter controls this (default: `0.5`).

### Doc-wise RoPE

This is the fix to the "train-on-short, fail-on-long" problem:

```python
# WRONG — global positional encoding (breaks at scale)
position_ids = torch.arange(total_tokens)  # 0 ... 100_000_000

# CORRECT — doc-wise independent RoPE
position_ids = []
for doc in documents:
    position_ids.append(torch.arange(len(doc)))   # always 0 ... doc_len
# Each document's positional IDs reset to 0 independently
```

For the active query context, use global RoPE offset by k:
```python
query_position_ids = torch.arange(k, k + query_len)
# Offset by k so the model perceives the query as a logical continuation
```

---

## Memory Interleave (Multi-Hop)

The iterative retrieve → generate → retrieve loop for complex reasoning:

1. Generate document IDs relevant to the query
2. Load those documents and append to context
3. Re-query with enriched context
4. Repeat until model emits `<end_of_retrieve>`
5. Generate the final answer

**Ablation result**: With Memory Interleave = 4.020, Without = 3.250 (−19.2%)

---

## Memory Parallel (100M Scale)

Tiered storage strategy:

| Component | Storage | Size (100M tokens) | Purpose |
|---|---|---|---|
| K̄ᴿ (routing keys) | GPU VRAM | ~56 GB | Fast cosine scoring |
| K̄, V̄ (content KVs) | CPU DRAM | ~113 GB | On-demand top-k fetch |

Only the top-k selected content KVs are transferred CPU → GPU per query, never the full 100M tokens.

---

## How This Differs from Supermemory

| Aspect | Omni-Memory (MSA) | Supermemory |
|---|---|---|
| Retrieval | Latent-space routing (in-model) | External embeddings (model-agnostic) |
| Optimisation | Joint routing + generation loss | Decoupled (no joint gradient flow) |
| Multi-hop | Iterative Memory Interleave | Single-shot |
| Positions | Doc-wise RoPE (scale-invariant) | N/A |
| State | Static files (zero writes per query) | Stateful DB (write-per-request) |
| Forgetting | None | Intentional decay |

---

## Source Files

| File | Stage | Purpose |
|---|---|---|
| `encode.py` | 1 | Offline corpus → compressed KV cache |
| `route.py` | 2 | Online query → top-k document selection |
| `generate.py` | 3 | Sparse attention → answer generation |
| `interleave.py` | — | Multi-hop Memory Interleave loop |
| `parallel.py` | — | Tiered GPU/CPU storage for 100M scale |
| `train.py` | — | End-to-end training with auxiliary loss |
| `rope_utils.py` | — | Doc-wise + global RoPE utilities |
| `config.py` | — | Central configuration |
| `server.py` | — | Stateless HTTP memory server |
