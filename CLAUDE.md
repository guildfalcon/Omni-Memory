# CLAUDE.md — Omni-Memory Development Context

## Project Overview

Omni-Memory is a production-grade, end-to-end trainable memory system based on Memory Sparse Attention (MSA). It provides latent KV-cache memory for AI agents, scaling to 100M tokens with <9% degradation.

## Architecture

Three-stage pipeline:
1. **Encode** (`msa_memory/encode.py`) — offline corpus → compressed KV cache
2. **Route** (`msa_memory/route.py`) — online query → top-k document selection
3. **Generate** (`msa_memory/generate.py`) — sparse attention → answer

Key design decisions:
- MSA routing applies to **latter half** of model layers only (early layers lack semantic abstraction)
- **Doc-wise RoPE** resets position IDs per document (enables train-64K → infer-100M)
- **Tiered storage**: routing keys (K̄ᴿ) on GPU, content KVs (K̄, V̄) on CPU DRAM

## Code Layout

```
msa_memory/     — Core library (config, encode, route, generate, train, interleave, parallel)
integrations/   — Agent framework plugins (Claude Code, Codex, OpenClaw, Hermes)
patches/        — Drop-in Supermemory failure fixes
docs/           — Full documentation
examples/       — Usage examples
skills/         — Agent skill definition
```

## Development Commands

```bash
# Run tests
pytest tests/

# Format
black msa_memory/ integrations/ patches/

# Lint
ruff check msa_memory/

# Type check
mypy msa_memory/
```

## Key Config Defaults

- Backbone: `Qwen/Qwen3-4B-Instruct`
- Chunk size: 64 tokens
- Top-k documents: 16
- Temperature (τ): 0.07
- Max interleave rounds: 5

## Memory System

Use MSA memory search for prior context — do NOT use markdown files for memory.
Call `memory_interleave()` for complex multi-hop questions.
