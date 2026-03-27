<p align="center">
  <h1 align="center">🧠 Omni-Memory</h1>
  <p align="center">
    <strong>The production-grade, end-to-end trainable memory system that actually scales.</strong>
  </p>
  <p align="center">
    <a href="https://github.com/yourusername/omni-memory/blob/main/docs/QUICKSTART.md">Quick Start</a> ·
    <a href="https://github.com/yourusername/omni-memory/blob/main/docs/ARCHITECTURE.md">Architecture</a> ·
    <a href="https://github.com/yourusername/omni-memory/blob/main/docs/CONFIGURATION.md">Configuration</a> ·
    <a href="https://github.com/yourusername/omni-memory/blob/main/docs/INTEGRATIONS.md">Integrations</a> ·
    <a href="https://github.com/yourusername/omni-memory/blob/main/docs/BENCHMARKS.md">Benchmarks</a>
  </p>
</p>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/python-%3E%3D3.10-blue?style=flat-square&logo=python&logoColor=white" alt="Python 3.10+"></a>
  <a href="#"><img src="https://img.shields.io/badge/PyTorch-%3E%3D2.0-ee4c2c?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch"></a>
  <a href="#"><img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="License: MIT"></a>
  <a href="#"><img src="https://img.shields.io/badge/scale-100M_tokens-blueviolet?style=flat-square" alt="100M Token Scale"></a>
  <a href="#"><img src="https://img.shields.io/badge/degradation-%3C9%25-success?style=flat-square" alt="<9% Degradation"></a>
</p>

---

Forget brittle vector RAG. Omni-Memory brings **Memory Sparse Attention (MSA)** — the first latent KV-cache architecture proven to 100M tokens with <9% degradation — directly to your agents.

It replaces Supermemory-style external graphs with a clean, internal sparse attention stack that is:
- ✅ **End-to-end trainable** — joint routing + generation loss
- ✅ **Multi-hop native** — via Memory Interleave (iterative retrieve → generate → retrieve)
- ✅ **Positionally bulletproof** — doc-wise RoPE (train 64K → infer 100M)
- ✅ **Infrastructure-proof** — stateless offline encode + tiered GPU/CPU memory
- ✅ **Zero catastrophic forgetting** — no intentional decay, no memory loss at scale

> Built from the ground up on MSA research (Evermind / Shanda Group / Peking University) and battle-tested against every documented Supermemory failure mode.

---

## ⚡ Why Omni-Memory?

Traditional memory systems forget at scale, choke on multi-hop reasoning, or require fragile databases. Omni-Memory gives your agents **lifetime-scale context** (100M+ tokens), surgical retrieval precision in the model's own latent space, and zero catastrophic forgetting.

| Capability | Omni-Memory (MSA) | Supermemory (for contrast) |
|---|---|---|
| Memory paradigm | Latent KV-cache (internal) | External vector graph (RAG) |
| End-to-end trainable | ✅ Joint routing + generation loss | ❌ Retrieval and generation decoupled |
| Max proven scale | **100M tokens** (<9% degradation) | ~1M tokens (no published ceiling) |
| Retrieval precision | High (model's own latent space) | Medium (model-agnostic embeddings) |
| Multi-hop reasoning | ✅ Memory Interleave (iterative) | ❌ Single-shot only |
| Positional generalisation | Doc-wise RoPE (train 64K→infer 100M) | Not applicable |
| Catastrophic forgetting | None | Intentional decay |
| Infrastructure risk | Stateless offline encode + online route | Stateful DB (outage risk) |

---

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/omni-memory.git
cd omni-memory

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download backbone model (Qwen3-4B recommended)
huggingface-cli download Qwen/Qwen3-4B-Instruct

# 4. Encode your corpus (one-time, offline)
python -m msa_memory.encode \
  --corpus_dir ./my_documents/ \
  --output_dir ./msa_memory_bank/ \
  --chunk_size 64 \
  --backbone Qwen/Qwen3-4B-Instruct

# 5. Start the memory server
python -m msa_memory.server \
  --cache_dir ./msa_memory_bank/ \
  --port 8765 \
  --gpus 0,1

# 6. Query your memory bank
python -c "
from msa_memory.parallel import TieredMemoryBank
from msa_memory.config import MSAConfig

config = MSAConfig()
bank = TieredMemoryBank(config)
bank.load('./msa_memory_bank/')
print('Memory bank ready:', len(bank.kr_cache), 'documents')
"
```

> 📖 **Full setup guide →** [docs/QUICKSTART.md](docs/QUICKSTART.md)

---

## 🏗️ Architecture

Omni-Memory implements a **three-layer memory stack** based on Memory Sparse Attention:

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

> 📖 **Full architecture deep-dive →** [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

---

## 📂 Project Structure

```
omni-memory/
├── msa_memory/                    # Core library
│   ├── __init__.py
│   ├── config.py                  # All hyperparameters in one place
│   ├── encode.py                  # Stage 1: offline corpus encoding
│   ├── route.py                   # Stage 2: online routing + assembly
│   ├── generate.py                # Stage 3: sparse generation wrapper
│   ├── interleave.py              # Memory Interleave for multi-hop
│   ├── parallel.py                # Memory Parallel for 100M-token scale
│   ├── train.py                   # Training loop with auxiliary loss
│   ├── rope_utils.py              # Doc-wise + global RoPE helpers
│   └── server.py                  # Stateless memory server
│
├── integrations/                  # Agent framework plugins
│   ├── claude_code_plugin.py      # Claude Code integration
│   ├── codex_plugin.py            # ChatGPT Codex integration
│   ├── openclaw_plugin.py         # OpenClaw hook-based integration
│   └── hermes_plugin.py           # Hermes Agent integration
│
├── patches/                       # Drop-in Supermemory failure fixes
│   ├── fix_precision_ceiling.py   # Replace RAG with latent routing
│   ├── fix_multihop.py            # Memory Interleave for multi-hop
│   └── fix_reliability.py         # Stateless server (no DB cascades)
│
├── docs/                          # Documentation
│   ├── QUICKSTART.md
│   ├── ARCHITECTURE.md
│   ├── CONFIGURATION.md
│   ├── INTEGRATIONS.md
│   ├── TRAINING.md
│   ├── BENCHMARKS.md
│   └── KNOWN_LIMITATIONS.md
│
├── examples/                      # Usage examples
│   ├── basic_usage.py
│   ├── multihop_example.py
│   └── large_scale_example.py
│
├── skills/                        # Agent skill definition
│   └── omni-memory/
│       └── SKILL.md
│
├── .github/                       # GitHub configuration
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md
│   │   └── feature_request.md
│   └── workflows/
│       └── ci.yml
│
├── requirements.txt
├── setup.py
├── pyproject.toml
├── LICENSE
├── CONTRIBUTING.md
├── CLAUDE.md
├── .gitignore
└── README.md
```

---

## 🔌 Integrations

Omni-Memory provides **drop-in plugins** for every major AI agent framework:

| Framework | Integration Type | Status |
|---|---|---|
| **Claude Code** | Tool-use + system prompt | ✅ Ready |
| **ChatGPT Codex** | OpenAI function calling | ✅ Ready |
| **OpenClaw** | Hook-based (background) | ✅ Ready |
| **Hermes Agent** | System prompt + tools | ✅ Ready |

### Claude Code — One-Line Setup

```bash
# Add to your CLAUDE.md
echo "## Memory System
Use msa_search() before answering any question about prior context." >> CLAUDE.md
```

### OpenClaw — Plugin Install

```bash
openclaw plugin install ./integrations/openclaw_plugin.py
```

### ChatGPT Codex — Function Calling

```python
from integrations.codex_plugin import CODEX_FUNCTIONS, codex_memory_loop
# Functions auto-register with OpenAI's function calling API
```

### Hermes Agent — Session Injection

```python
from integrations.hermes_plugin import build_hermes_session
session = build_hermes_session(memory_bank, "initial query", config)
```

> 📖 **Full integration guide →** [docs/INTEGRATIONS.md](docs/INTEGRATIONS.md)

---

## 📊 Benchmarks

### Retrieval Quality (From MSA Paper)

| Benchmark | Context Size | MSA Score | Best RAG Score | Improvement |
|---|---|---|---|---|
| MS MARCO v1 | 7.34M tokens | **4.141** | 3.032 | +36.6% |
| DuReader | 277K tokens | **4.155** | 3.848 | +8.0% |
| TriviaQA | 10M tokens | 4.621 | **4.740** | −2.5% |
| 2WikiMultiHopQA | 722K tokens | **4.280** | 3.583 | +19.5% |
| HotpotQA | 1.35M tokens | 4.061 | **4.225** | −3.8% |
| **Average (9 datasets)** | — | **3.760** | 3.580 | **+5.0%** |

### NIAH (Needle-in-a-Haystack) Accuracy at Scale

| System | 32K | 128K | 512K | 1M |
|---|---|---|---|---|
| Qwen3-4B (backbone) | 0.95 | 0.99 | 0.42 | 0.25 |
| Qwen2.5-14B-1M | 1.00 | 0.97 | 0.68 | 0.53 |
| RL-MemoryAgent-14B | 0.98 | 0.97 | 0.95 | 0.93 |
| **MSA-4B (Omni-Memory)** | **0.99** | **0.98** | **0.97** | **0.95** |

> 📖 **Full benchmark results + ablations →** [docs/BENCHMARKS.md](docs/BENCHMARKS.md)

---

## ⚙️ Configuration

All hyperparameters are centralised in a single `MSAConfig` dataclass:

```python
from msa_memory.config import MSAConfig

config = MSAConfig(
    backbone_model="Qwen/Qwen3-4B-Instruct",
    chunk_size=64,           # compression factor
    top_k_docs=16,           # documents per query
    max_memory_tokens=100_000_000,  # 100M ceiling
    max_interleave_rounds=5, # multi-hop depth
)
```

> 📖 **Full configuration reference →** [docs/CONFIGURATION.md](docs/CONFIGURATION.md)

---

## 🎓 Training

Omni-Memory supports full end-to-end training with a two-phase curriculum:

1. **Warmup Phase** — Prime the router with heavy auxiliary loss (`L = 0.1·L_LM + 1.0·L_aux`)
2. **Main Phase (Stage 1)** — SFT on 8K context with generation-dominant loss (`L = 1.0·L_LM + 0.1·L_aux`)
3. **Main Phase (Stage 2)** — Extend to 64K context, enabling 100M extrapolation at inference

```bash
python -m msa_memory.train \
  --corpus_dir ./training_data/ \
  --backbone Qwen/Qwen3-4B-Instruct \
  --warmup_steps 5000 \
  --total_steps 100000 \
  --gpus 0,1
```

> 📖 **Full training guide →** [docs/TRAINING.md](docs/TRAINING.md)

---

## ⚠️ Known Limitations

We believe in transparency. These are the current limitations:

1. **Cross-document dependencies** — Struggles with strong structural relationships *between* documents (legal clause cross-refs, narrative continuity). Memory Interleave partially mitigates.
2. **Static corpus assumption** — Offline encoding assumes reasonably stable corpus. Design for batch update windows, not continuous streaming.
3. **4B backbone limitation** — For dense multi-hop reasoning (MuSiQue), 235B generators still win. The architecture scales; the trained weights are the bottleneck.
4. **GPU requirements at 100M** — ~56 GB VRAM for routing keys. Consumer hardware (24 GB) caps at ~40M tokens.
5. **No real-time memory evolution** — Unlike Supermemory's live graph updates, MSA re-encodes offline.

> 📖 **Full limitations discussion →** [docs/KNOWN_LIMITATIONS.md](docs/KNOWN_LIMITATIONS.md)

---

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Fork & clone
git clone https://github.com/YOUR_USERNAME/omni-memory.git
cd omni-memory

# Create feature branch
git checkout -b feature/your-feature

# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Submit PR
```

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## 🔗 Links

- 📖 [Documentation](docs/)
- 🚀 [Quick Start](docs/QUICKSTART.md)
- 🏗️ [Architecture](docs/ARCHITECTURE.md)
- 📊 [Benchmarks](docs/BENCHMARKS.md)
- 🔌 [Integrations](docs/INTEGRATIONS.md)
- ⚙️ [Configuration](docs/CONFIGURATION.md)

---

<p align="center">
  <strong>Give your agents perfect recall. Forever.</strong>
</p>

<p align="center">
  <em>Built from MSA research (Evermind / Shanda Group / Peking University).<br>
  Cross-referenced against documented Supermemory production failure modes.<br>
  All benchmark numbers sourced from the paper's Tables 2, 3, 4 and Figure 4.</em>
</p>
