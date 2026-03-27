# Known Limitations

> Be transparent about these. Overpromising causes the same trust erosion that hurt Supermemory.
>
> — From Paper Section 7

---

## 1. Tightly Coupled Cross-Document Dependencies

MSA struggles when the answer requires modelling strong structural relationships *between* documents — for example, legal clause cross-references, narrative continuity in long fiction, or dependency chains in large codebases.

**Why**: Each document is encoded independently with doc-wise RoPE. The model doesn't see cross-document positional relationships during encoding.

**Mitigation**: Memory Interleave partially mitigates this by iteratively accumulating context across documents. For truly structural dependencies, consider pre-processing related documents into a single concatenated unit.

---

## 2. Static Corpus Assumption

The offline encoding pipeline assumes the corpus is reasonably stable. It is not designed for high-velocity streaming data (e.g., chat messages arriving every second).

**Why**: Re-encoding the full corpus takes minutes to hours depending on size. There is no incremental encoding in the current implementation.

**Mitigation**: Design for batch update windows (e.g., nightly re-encode). For truly real-time needs, consider a hybrid approach: MSA for the stable knowledge base + a lightweight live buffer for recent messages.

---

## 3. 4B Backbone Limitation

The published MSA model uses a 4B parameter backbone (Qwen3-4B-Instruct). For dense multi-hop reasoning tasks like MuSiQue, 235B generators still win.

**Why**: The architecture scales — the trained weights are the bottleneck. 58× fewer parameters means less intrinsic reasoning capacity, regardless of memory quality.

**Evidence**: MSA-4B = 2.211 vs KaLMv2+Qwen3-235B = 2.647 on MuSiQue.

**Future**: The architecture is backbone-agnostic. Training with larger backbones (8B, 14B, 70B) should improve dense reasoning scores.

---

## 4. GPU Requirements at 100M Scale

The tiered storage strategy still requires ~56 GB VRAM for routing keys at 100M tokens. This exceeds consumer hardware limits.

| Token Count | VRAM Required | Hardware |
|---|---|---|
| 10M | ~5.6 GB | Any modern GPU |
| 40M | ~22.4 GB | RTX 4090 / A6000 |
| 100M | ~56 GB | A100 / A800 / H100 |

**Mitigation**: On consumer hardware (single 24 GB GPU), cap at ~40M tokens. Use the `max_memory_tokens` config parameter.

---

## 5. No Real-Time Memory Evolution

Unlike Supermemory's graph which updates live, MSA re-encodes offline. New information is not available until the next encoding batch completes.

**Why**: Encoding requires a full forward pass through the backbone model per document. This is computationally intensive and not suitable for per-message granularity.

**Mitigation**: Implement a dual-buffer strategy:
- **Stable memory**: MSA-encoded corpus (re-encoded on schedule)
- **Live buffer**: Small ring buffer of recent messages (simple text search)
- **Merge**: Periodically absorb the live buffer into the stable memory via re-encoding
