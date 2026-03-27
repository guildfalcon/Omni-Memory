# Configuration Reference

All Omni-Memory hyperparameters are centralised in a single `MSAConfig` dataclass in [`msa_memory/config.py`](../msa_memory/config.py).

---

## Quick Start

```python
from msa_memory.config import MSAConfig

# Default configuration (Qwen3-4B, 100M token ceiling)
config = MSAConfig()

# Custom configuration
config = MSAConfig(
    backbone_model="meta-llama/Llama-3-8B-Instruct",
    chunk_size=128,
    top_k_docs=32,
    max_memory_tokens=10_000_000,
)
```

---

## Parameter Reference

### Core Architecture

| Parameter | Type | Default | Description |
|---|---|---|---|
| `backbone_model` | `str` | `"Qwen/Qwen3-4B-Instruct"` | HuggingFace model ID or local path |
| `chunk_size` | `int` | `64` | Tokens per compression chunk (P). Smaller = finer retrieval, more memory |
| `top_k_docs` | `int` | `16` | Documents retrieved per query. Higher = better recall, slower generation |
| `router_layers_fraction` | `float` | `0.5` | Fraction of layers using MSA routing (latter half only) |
| `head_dim` | `int` | `128` | Dimension per attention head (must match backbone) |
| `num_heads` | `int` | `8` | Number of attention heads (must match backbone) |

### Scale Limits

| Parameter | Type | Default | Description |
|---|---|---|---|
| `max_memory_tokens` | `int` | `100,000,000` | Maximum tokens in memory bank |
| `max_query_tokens` | `int` | `4,096` | Maximum tokens per query |
| `max_answer_tokens` | `int` | `1,024` | Maximum tokens per generated answer |

### Environment-Specific Scale Limits

| Hardware | Recommended `max_memory_tokens` |
|---|---|
| 2×A800 (160 GB VRAM) | `100,000,000` (100M) |
| 1×A100 (80 GB VRAM) | `40,000,000` (40M) |
| 1×RTX 4090 (24 GB VRAM) | `10,000,000` (10M) |
| CPU-only | `10,000,000` (10M) |

### Training

| Parameter | Type | Default | Description |
|---|---|---|---|
| `warmup_lm_weight` | `float` | `0.1` | LM loss weight during warmup |
| `warmup_aux_weight` | `float` | `1.0` | Auxiliary loss weight during warmup |
| `main_lm_weight` | `float` | `1.0` | LM loss weight during main training |
| `main_aux_weight` | `float` | `0.1` | Auxiliary loss weight during main training |
| `warmup_lr` | `float` | `1e-4` | Learning rate during warmup |
| `main_lr` | `float` | `6e-6` | Learning rate during main training |
| `warmup_steps` | `int` | `5,000` | Number of warmup steps |
| `temperature` | `float` | `0.07` | Temperature τ for contrastive loss |

### Tiered Storage (Memory Parallel)

| Parameter | Type | Default | Description |
|---|---|---|---|
| `routing_keys_device` | `str` | `"cuda"` | Device for routing keys (K̄ᴿ) |
| `content_kvs_device` | `str` | `"cpu"` | Device for content KVs (K̄, V̄) |
| `async_prefetch` | `bool` | `True` | Enable async CPU→GPU transfer for top-k |

### Memory Interleave (Multi-Hop)

| Parameter | Type | Default | Description |
|---|---|---|---|
| `max_interleave_rounds` | `int` | `5` | Maximum retrieve→generate cycles |
| `interleave_delimiter` | `str` | `"<end_of_retrieve>"` | Token signalling retrieval completion |

---

## Common Configuration Profiles

### Development (CPU-only, small corpus)

```python
config = MSAConfig(
    backbone_model="Qwen/Qwen3-4B-Instruct",
    max_memory_tokens=1_000_000,
    routing_keys_device="cpu",
    content_kvs_device="cpu",
    top_k_docs=8,
)
```

### Production (Single A100)

```python
config = MSAConfig(
    backbone_model="Qwen/Qwen3-4B-Instruct",
    max_memory_tokens=40_000_000,
    routing_keys_device="cuda:0",
    content_kvs_device="cpu",
    top_k_docs=16,
)
```

### Maximum Scale (2×A800)

```python
config = MSAConfig(
    backbone_model="Qwen/Qwen3-4B-Instruct",
    max_memory_tokens=100_000_000,
    routing_keys_device="cuda",
    content_kvs_device="cpu",
    top_k_docs=32,
    async_prefetch=True,
)
```

---

## Tuning Guidelines

### `chunk_size` — Compression vs. Precision Trade-off

- **Lower values (16–32)**: Finer-grained retrieval, more memory usage, better precision on short documents
- **Default (64)**: Balanced — recommended starting point
- **Higher values (128–256)**: More compression, less memory, may lose fine details

### `top_k_docs` — Recall vs. Latency Trade-off

- **Lower values (4–8)**: Faster generation, may miss relevant documents
- **Default (16)**: Good balance for most use cases
- **Higher values (32–64)**: Better recall, slower generation due to larger context

### `temperature` — Routing Sharpness

- **Lower values (0.01–0.05)**: Sharper routing decisions, risk of routing collapse
- **Default (0.07)**: Balanced — used in the paper
- **Higher values (0.1–0.2)**: Softer routing, more exploration during training
