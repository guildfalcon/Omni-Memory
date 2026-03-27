# Training Guide

Omni-Memory supports full end-to-end training with a two-phase curriculum. This guide covers the complete training pipeline.

---

## Why Train?

Unlike decoupled RAG systems, MSA is **end-to-end trainable**. This means:

- The router learns which documents are relevant **jointly** with the generator
- Routing decisions are supervised by the auxiliary contrastive loss
- The model learns to route AND generate in the same optimisation step

**Ablation**: Removing continual pre-training causes a **36.2%** average performance drop. Training is not optional for production quality.

---

## Training Phases

### Phase 1: Warmup (Priming the Router)

```
Loss = 0.1 · L_LM + 1.0 · L_aux
LR   = 1e-4
Context = 8K tokens
```

**Purpose**: Rapidly align the Router Projectors (K̄ᴿ) before the generator dominates. The heavy auxiliary weight forces the routing layer to learn correct document selection.

### Phase 2: Main Training — Stage 1 (8K Context)

```
Loss = 1.0 · L_LM + 0.1 · L_aux
LR   = 6e-6
Context = 8K tokens
```

**Purpose**: Standard supervised fine-tuning. The generator now dominates, while the routing signal is maintained at low weight.

### Phase 3: Main Training — Stage 2 (64K Context)

```
Loss = 1.0 · L_LM + 0.1 · L_aux
LR   = 6e-6
Context = 64K tokens
```

**Purpose**: Extend context to 64K, which enables extrapolation to 100M tokens at inference. This is the critical step for scale.

**Ablation**: MSA-S2 (both stages) beats MSA-S1 (stage 1 only) by 7.6% average, and by 29.5% on MS MARCO.

---

## The Auxiliary Contrastive Loss

The core innovation that makes routing trainable:

```
Laux = -1/|P| · Σᵢ log( exp(sᵢ⁺/τ) / (exp(sᵢ⁺/τ) + Σⱼ exp(sᵢⱼ⁻/τ)) )
```

Where:
- `sᵢ⁺` = routing score for the **correct** document
- `sᵢⱼ⁻` = routing scores for **wrong** documents
- `τ` = temperature (default: 0.07)

This loss directly supervises which documents the router selects, creating a gradient path from routing decisions to model weights.

---

## Running Training

### Command Line

```bash
python -m msa_memory.train \
  --corpus_dir ./training_data/ \
  --backbone Qwen/Qwen3-4B-Instruct \
  --warmup_steps 5000 \
  --total_steps 100000 \
  --gpus 0,1
```

### Programmatic

```python
from msa_memory.config import MSAConfig
from msa_memory.train import (
    auxiliary_routing_loss,
    combined_loss,
    curriculum_training_schedule,
)

config = MSAConfig()
schedule = curriculum_training_schedule(total_steps=100000)

for phase in schedule:
    print(f"Phase: {phase['phase']}")
    print(f"  Steps: {phase['steps']}")
    print(f"  LR: {phase['lr']}")
    print(f"  Context: {phase['context_len']}")
    print(f"  Weights: {phase['loss_weights']}")
```

---

## Data Preparation

Training requires documents annotated with positive/negative pairs:

```json
{
  "query": "What is the capital of France?",
  "positive_doc_ids": [42],
  "negative_doc_ids": [7, 15, 33, 89, 102],
  "answer": "The capital of France is Paris."
}
```

### Tips

1. **Positive documents** should contain the evidence needed to answer the query
2. **Hard negatives** (topically related but wrong) are more valuable than random negatives
3. Aim for at least 5 negatives per positive
4. Use diverse query types: factoid, multi-hop, preference, decision recall

---

## Hardware Requirements

| Scale | GPU | Training Time (est.) |
|---|---|---|
| Small (10K steps) | 1× RTX 4090 | ~4 hours |
| Medium (50K steps) | 1× A100 | ~12 hours |
| Full (100K steps) | 2× A800 | ~24 hours |

---

## Critical Rules

> **Never skip continual pre-training.** Ablation shows −36.2% without it.
>
> **Never skip original text injection.** Ablation shows −41.5% without it.
>
> **Always use two-stage curriculum.** Single-stage loses −7.6% average.
