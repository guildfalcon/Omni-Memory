# Omni-Memory — GitHub Repository Walkthrough

## What Was Built

Packaged the MSA Memory System SKILL.md into a **full-fledged GitHub repository** at `c:\Omni-Memory`, modeled after the [supermemory](https://github.com/supermemoryai/supermemory) repo structure.

## Repository Structure (47 files)

### Core Library — `msa_memory/` (10 files)
| File | Purpose |
|---|---|
| [__init__.py](file:///c:/Omni-Memory/msa_memory/__init__.py) | Package with lazy imports for all public APIs |
| [config.py](file:///c:/Omni-Memory/msa_memory/config.py) | Central `MSAConfig` dataclass with all hyperparameters |
| [encode.py](file:///c:/Omni-Memory/msa_memory/encode.py) | **Stage 1**: Offline corpus → compressed KV cache (K̄, V̄, K̄ᴿ) |
| [route.py](file:///c:/Omni-Memory/msa_memory/route.py) | **Stage 2**: Online query → top-k document selection |
| [generate.py](file:///c:/Omni-Memory/msa_memory/generate.py) | **Stage 3**: Sparse attention → answer generation |
| [interleave.py](file:///c:/Omni-Memory/msa_memory/interleave.py) | Memory Interleave for multi-hop reasoning |
| [parallel.py](file:///c:/Omni-Memory/msa_memory/parallel.py) | Tiered GPU/CPU storage for 100M-token scale |
| [train.py](file:///c:/Omni-Memory/msa_memory/train.py) | End-to-end training with auxiliary contrastive loss |
| [rope_utils.py](file:///c:/Omni-Memory/msa_memory/rope_utils.py) | Doc-wise RoPE for positional generalisation |
| [server.py](file:///c:/Omni-Memory/msa_memory/server.py) | Stateless HTTP memory server (zero DB writes) |

### Integrations — `integrations/` (4 plugins)
| Plugin | Framework | Type |
|---|---|---|
| [claude_code_plugin.py](file:///c:/Omni-Memory/integrations/claude_code_plugin.py) | Claude Code | Tool-use + system prompt |
| [codex_plugin.py](file:///c:/Omni-Memory/integrations/codex_plugin.py) | ChatGPT Codex | OpenAI function calling |
| [openclaw_plugin.py](file:///c:/Omni-Memory/integrations/openclaw_plugin.py) | OpenClaw | Hook-based background memory |
| [hermes_plugin.py](file:///c:/Omni-Memory/integrations/hermes_plugin.py) | Hermes Agent | System prompt + tools |

### Patches — `patches/` (3 fixes)
| Patch | Supermemory Failure Fixed |
|---|---|
| [fix_precision_ceiling.py](file:///c:/Omni-Memory/patches/fix_precision_ceiling.py) | External embeddings → latent routing |
| [fix_multihop.py](file:///c:/Omni-Memory/patches/fix_multihop.py) | Single-shot → Memory Interleave |
| [fix_reliability.py](file:///c:/Omni-Memory/patches/fix_reliability.py) | Stateful DB → stateless file cache |

### Documentation — `docs/` (7 guides)
| Doc | Contents |
|---|---|
| [QUICKSTART.md](file:///c:/Omni-Memory/docs/QUICKSTART.md) | 5-minute setup, install, encode, query |
| [ARCHITECTURE.md](file:///c:/Omni-Memory/docs/ARCHITECTURE.md) | Three-stage pipeline, core math, RoPE fix |
| [CONFIGURATION.md](file:///c:/Omni-Memory/docs/CONFIGURATION.md) | All parameters, profiles, tuning guidelines |
| [INTEGRATIONS.md](file:///c:/Omni-Memory/docs/INTEGRATIONS.md) | Setup for all 4 agent frameworks |
| [TRAINING.md](file:///c:/Omni-Memory/docs/TRAINING.md) | Two-phase curriculum, auxiliary loss, data prep |
| [BENCHMARKS.md](file:///c:/Omni-Memory/docs/BENCHMARKS.md) | Paper results, ablations, honest failure cases |
| [KNOWN_LIMITATIONS.md](file:///c:/Omni-Memory/docs/KNOWN_LIMITATIONS.md) | 5 limitations with mitigations |

### Examples — `examples/` (3 scripts)
- [basic_usage.py](file:///c:/Omni-Memory/examples/basic_usage.py) — Load and query a memory bank
- [multihop_example.py](file:///c:/Omni-Memory/examples/multihop_example.py) — Memory Interleave walkthrough
- [large_scale_example.py](file:///c:/Omni-Memory/examples/large_scale_example.py) — 100M-token tiered storage

### Tests — `tests/` (5 test files, 37 tests)
- [test_config.py](file:///c:/Omni-Memory/tests/test_config.py) — Config defaults and overrides
- [test_encode.py](file:///c:/Omni-Memory/tests/test_encode.py) — chunk_mean_pool shapes and correctness
- [test_rope_utils.py](file:///c:/Omni-Memory/tests/test_rope_utils.py) — Doc-wise and query position IDs
- [test_train.py](file:///c:/Omni-Memory/tests/test_train.py) — Auxiliary loss, combined loss, curriculum
- [test_interleave.py](file:///c:/Omni-Memory/tests/test_interleave.py) — Doc ID parsing, answer cleaning

### Project Config & GitHub
| File | Purpose |
|---|---|
| [README.md](file:///c:/Omni-Memory/README.md) | Hero README with badges, tables, quick start |
| [CONTRIBUTING.md](file:///c:/Omni-Memory/CONTRIBUTING.md) | Contributing guidelines and dev workflow |
| [CLAUDE.md](file:///c:/Omni-Memory/CLAUDE.md) | Developer context for Claude Code |
| [LICENSE](file:///c:/Omni-Memory/LICENSE) | MIT License |
| [pyproject.toml](file:///c:/Omni-Memory/pyproject.toml) | Package metadata, deps, tool configs |
| [setup.py](file:///c:/Omni-Memory/setup.py) | Legacy setup for backwards compat |
| [requirements.txt](file:///c:/Omni-Memory/requirements.txt) | Core dependencies |
| [requirements-dev.txt](file:///c:/Omni-Memory/requirements-dev.txt) | Dev/test dependencies |
| [.gitignore](file:///c:/Omni-Memory/.gitignore) | Python, PyTorch, IDE, generated files |
| [.github/workflows/ci.yml](file:///c:/Omni-Memory/.github/workflows/ci.yml) | CI: lint + test on Python 3.10/3.11/3.12 |
| [.github/ISSUE_TEMPLATE/](file:///c:/Omni-Memory/.github/ISSUE_TEMPLATE/) | Bug report + feature request templates |
| [skills/omni-memory/SKILL.md](file:///c:/Omni-Memory/skills/omni-memory/SKILL.md) | Original skill file (preserved) |

## Verification

- ✅ **37 tests passing** across config, encode, rope_utils, train, and interleave modules
- ✅ Git repository initialized with `.gitignore` properly configured
- ✅ All source code extracted from SKILL.md into proper Python modules
- ✅ Original SKILL.md preserved in `skills/omni-memory/`
- ✅ User's readme.md content incorporated into the main README.md

## Inspiration from Supermemory Repo

Adopted the following patterns from [supermemory](https://github.com/supermemoryai/supermemory):
- Professional README with badges, feature comparison table, quick start
- `CLAUDE.md` for AI agent development context
- `CONTRIBUTING.md` with clear setup instructions
- `skills/` directory for agent skill definitions
- MIT License
- GitHub Actions CI pipeline
- Issue templates (bug report + feature request)
