# Benchmarks

All benchmark numbers are sourced directly from the MSA paper's Tables 2, 3, 4 and Figure 4.

---

## Retrieval Quality (9 Datasets)

| Benchmark | Context Size | MSA Score | Best RAG Score | Delta |
|---|---|---|---|---|
| MS MARCO v1 | 7.34M tokens | **4.141** | 3.032 | **+36.6%** |
| DuReader | 277K tokens | **4.155** | 3.848 | **+8.0%** |
| TriviaQA | 10M tokens | 4.621 | **4.740** | −2.5% |
| 2WikiMultiHopQA | 722K tokens | **4.280** | 3.583 | **+19.5%** |
| HotpotQA | 1.35M tokens | 4.061 | **4.225** | −3.8% |
| MuSiQue | — | 2.211 | **2.647** | −16.5% |
| **Average (9 datasets)** | — | **3.760** | 3.580 | **+5.0%** |

---

## NIAH (Needle-in-a-Haystack) Accuracy at Scale

| System | 32K | 128K | 512K | 1M |
|---|---|---|---|---|
| Qwen3-4B (backbone alone) | 0.95 | 0.99 | 0.42 | 0.25 |
| Qwen2.5-14B-1M | 1.00 | 0.97 | 0.68 | 0.53 |
| RL-MemoryAgent-14B | 0.98 | 0.97 | 0.95 | 0.93 |
| **MSA-4B (Omni-Memory)** | **0.99** | **0.98** | **0.97** | **0.95** |

Key insight: MSA-4B (4 billion parameters) outperforms RL-MemoryAgent-14B (14 billion parameters) at 1M token scale, demonstrating that the architecture — not just parameter count — drives scale performance.

---

## When MSA Loses to RAG

We believe in transparency. These are the cases where traditional RAG outperforms MSA:

### 1. MuSiQue (Dense Multi-Hop Reasoning)
- **MSA-4B**: 2.211
- **KaLMv2 + Qwen3-235B**: 2.647

The 235B parameter generator has vastly stronger intrinsic reasoning. MSA's 4B backbone cannot compensate for 58× fewer parameters. The architecture is sound; the trained weights are the bottleneck.

### 2. TriviaQA (Factoid Retrieval)
- **MSA**: 4.621
- **Best RAG**: 4.740

Marginal loss. Best-of-breed RAG with large generators still wins on factoid retrieval where the answer is a single verbatim span.

### 3. Dynamic Knowledge Updates
MSA requires offline re-encoding when corpus changes. For real-time streaming memory (new messages every second), graph-based live updates remain more practical.

---

## Ablation Results — What Breaks What

| Component Removed | Avg Score | Drop from Full |
|---|---|---|
| Full MSA-S2 (baseline) | **3.976** | — |
| − Stage 2 curriculum (use MSA-S1) | 3.694 | −7.6% |
| − Memory Interleave | 3.497 | −11.9% |
| − Continual pre-training | 2.537 | **−36.2%** |
| − Original document text injection | 2.325 | **−41.5%** |

### Critical Takeaways

1. **Never skip continual pre-training** — Single largest performance factor (−36.2%)
2. **Never skip original text injection** — Second largest factor (−41.5%)
3. **Memory Interleave is essential** for multi-hop tasks (−11.9%)
4. **Two-stage curriculum matters** — Stage 2 (64K extension) adds 7.6% on average

---

## Memory Budget at Scale

| Token Count | Routing Keys (VRAM) | Content KVs (DRAM) | Total |
|---|---|---|---|
| 1M | ~0.56 GB | ~1.13 GB | ~1.69 GB |
| 10M | ~5.6 GB | ~11.3 GB | ~16.9 GB |
| 40M | ~22.4 GB | ~45.2 GB | ~67.6 GB |
| 100M | ~56 GB | ~113 GB | ~169 GB |

---

## Reproducing These Results

1. Use the same backbone: `Qwen/Qwen3-4B-Instruct`
2. Follow the two-stage curriculum exactly (see [TRAINING.md](TRAINING.md))
3. Use chunk_size=64, top_k_docs=16 (paper defaults)
4. Train for ≥100K steps with both warmup and main phases
5. Evaluate on the same benchmark datasets at the same context lengths
