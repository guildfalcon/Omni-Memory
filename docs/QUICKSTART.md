# Quick Start Guide

Get Omni-Memory running in 5 minutes.

## Prerequisites

- **Python** ≥ 3.10
- **PyTorch** ≥ 2.0
- **GPU** (recommended): NVIDIA GPU with ≥16 GB VRAM for real-time routing
- **CPU-only**: Works up to ~10M tokens without GPU

## Installation

### From Source (Recommended)

```bash
git clone https://github.com/yourusername/omni-memory.git
cd omni-memory
pip install -r requirements.txt
```

### Via pip

```bash
pip install omni-memory
```

## Step 1: Download a Backbone Model

Omni-Memory uses a HuggingFace transformer as its backbone. We recommend **Qwen3-4B-Instruct** (the same backbone used in the MSA paper):

```bash
huggingface-cli download Qwen/Qwen3-4B-Instruct
```

Other compatible backbones:
- `meta-llama/Llama-3-8B-Instruct`
- `mistralai/Mistral-7B-Instruct-v0.3`
- Any HuggingFace model with standard attention layers

## Step 2: Prepare Your Corpus

Organize your documents as text files in a directory:

```
my_documents/
├── meeting_notes_2024.txt
├── codebase_decisions.md
├── user_preferences.txt
├── project_architecture.md
└── ... (any .txt, .md, .py, .json, .html files)
```

## Step 3: Encode the Corpus (One-Time)

Run the offline encoder to transform your documents into compressed KV cache:

```bash
python -m msa_memory.encode \
  --corpus_dir ./my_documents/ \
  --output_dir ./msa_memory_bank/ \
  --chunk_size 64 \
  --backbone Qwen/Qwen3-4B-Instruct
```

This produces:
- `routing_keys.pt` — Compressed routing keys (goes to GPU VRAM)
- `content_k.pt` — Compressed standard keys (stays in CPU DRAM)
- `content_v.pt` — Compressed standard values (stays in CPU DRAM)
- `doc_index.json` — Document metadata for Memory Interleave

> **⏱ Encoding time**: Varies by corpus size. ~1000 documents takes ~5 minutes on an A100.

## Step 4: Start the Memory Server

```bash
python -m msa_memory.server \
  --cache_dir ./msa_memory_bank/ \
  --port 8765 \
  --gpus 0
```

The server exposes:
- `GET /health` — Health check
- `POST /search` — Search the memory bank
- `POST /search/multihop` — Multi-hop Memory Interleave search

## Step 5: Query Your Memory

### Via Python API

```python
from msa_memory.config import MSAConfig
from msa_memory.parallel import TieredMemoryBank

config = MSAConfig()
bank = TieredMemoryBank(config)
bank.load("./msa_memory_bank/")

print(f"Memory bank ready: {len(bank.kr_cache)} documents")
```

### Via HTTP API

```bash
# Health check
curl http://localhost:8765/health

# Search
curl -X POST http://localhost:8765/search \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the user preferences for code style?", "top_k": 5}'
```

## Step 6: Integrate with Your Agent

Choose your agent framework:

### Claude Code
```bash
echo "## Memory System
Use msa_search() before answering any question about prior context." >> CLAUDE.md
```

### OpenClaw
```bash
openclaw plugin install ./integrations/openclaw_plugin.py
```

### ChatGPT Codex
```python
from integrations.codex_plugin import CODEX_FUNCTIONS
# Functions auto-register with OpenAI's function calling API
```

### Hermes Agent
```python
from integrations.hermes_plugin import build_hermes_session
session = build_hermes_session(bank, "initial query", config)
```

---

## Next Steps

- 📐 [Architecture Deep-Dive](ARCHITECTURE.md) — Understand the three-layer memory stack
- ⚙️ [Configuration Reference](CONFIGURATION.md) — Tune all hyperparameters
- 🎓 [Training Guide](TRAINING.md) — Train end-to-end on your own data
- 📊 [Benchmarks](BENCHMARKS.md) — Performance expectations and ablations
- 🔌 [Full Integration Guide](INTEGRATIONS.md) — Detailed plugin setup
