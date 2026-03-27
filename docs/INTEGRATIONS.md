# Integrations Guide

Omni-Memory provides drop-in plugins for every major AI agent framework. Each integration handles the connection between your agent and the MSA memory system.

---

## Supported Frameworks

| Framework | Integration Type | File | Status |
|---|---|---|---|
| **Claude Code** | Tool-use + system prompt | `integrations/claude_code_plugin.py` | ✅ Ready |
| **ChatGPT Codex** | OpenAI function calling | `integrations/codex_plugin.py` | ✅ Ready |
| **OpenClaw** | Hook-based (background) | `integrations/openclaw_plugin.py` | ✅ Ready |
| **Hermes Agent** | System prompt + tools | `integrations/hermes_plugin.py` | ✅ Ready |

---

## Claude Code

### How It Works

The Claude Code integration registers three tools via Claude's `tool_use` format:

1. **`memory_search`** — Search the MSA memory bank using latent-space routing
2. **`memory_store`** — Add new documents to the memory bank
3. **`memory_interleave`** — Run multi-hop Memory Interleave for complex questions

### Setup

**Option 1: Add to CLAUDE.md**

Copy the system prompt from the plugin into your `CLAUDE.md`:

```bash
echo "## Memory System
Use msa_search() before answering any question about prior context.
Memory is stored in latent KV-cache format, not markdown files.
Call memory_interleave() for complex multi-hop questions." >> CLAUDE.md
```

**Option 2: Register tools programmatically**

```python
from integrations.claude_code_plugin import register_claude_code_tools

tools = register_claude_code_tools()
# Returns list of tool definitions in Claude's tool_use format
```

### What Changes

| Before (CLAUDE.md files) | After (MSA Memory) |
|---|---|
| 54.2% accuracy on MemoryBench | 85.9%+ accuracy |
| Text files on disk | Latent KV-cache in model's own space |
| Manual search through markdown | Automatic latent-space routing |
| No multi-hop reasoning | Full Memory Interleave support |

---

## ChatGPT Codex

### How It Works

The Codex integration provides OpenAI function calling compatible definitions:

1. **`msa_memory_search`** — Search with optional multi-hop
2. **`msa_memory_store`** — Store new information with tags

### Setup

```python
from integrations.codex_plugin import CODEX_FUNCTIONS, codex_memory_loop

# Use CODEX_FUNCTIONS directly with OpenAI API
response = openai.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    functions=CODEX_FUNCTIONS,
    function_call="auto",
)

# Or use the full memory loop helper
answer = codex_memory_loop(
    user_message="What patterns did we discuss for the auth module?",
    memory_bank=bank,
    config=config,
    conversation_history=history,
)
```

---

## OpenClaw

### How It Works

The OpenClaw integration uses **hooks** instead of tool calls. This is a fundamental design difference:

- **Tool-call model** (Supermemory default): Agent must explicitly call `memory_search()` → slow, uses tokens, agent often forgets to call it
- **Hook model** (MSA): Memory is injected automatically before every generation → fast, always-on, zero token overhead

### Hooks

| Hook | Trigger | What It Does |
|---|---|---|
| `on_message_received` | Every message | Background encoding (non-blocking) |
| `on_response_generated` | After response | Implicit memory save |
| `on_session_start` | Session start | Load memory bank |
| `before_generation` | Before each turn | Inject top-k context into prompt |

### Setup

```bash
openclaw plugin install ./integrations/openclaw_plugin.py
```

### Configuration

```python
OPENCLAW_PLUGIN_CONFIG = {
    "name": "msa-memory",
    "hooks": {
        "on_message_received": "msa_ingest_hook",
        "on_response_generated": "msa_store_hook",
        "on_session_start": "msa_load_hook",
        "before_generation": "msa_inject_hook",
    },
    "replaces": ["MEMORY.md", "memory/YYYY-MM-DD.md", "qmd_memory_plugin"],
}
```

---

## Hermes Agent

### How It Works

Hermes uses a structured `<|im_start|>system` prompt with tool definitions. The MSA plugin:

1. Injects memory stats and pre-fetched context into the system prompt
2. Provides three tool definitions: `msa_search`, `msa_store`, `msa_multihop`

### Setup

```python
from integrations.hermes_plugin import build_hermes_session

session_prompt = build_hermes_session(
    memory_bank=bank,
    initial_query="What are the user's code style preferences?",
    config=config,
)
# session_prompt now contains:
#   - System prompt with memory stats
#   - Pre-fetched top-k context for the initial query
#   - Tool definitions in Hermes format
```

### Tool Definitions

| Tool | Purpose | When to Use |
|---|---|---|
| `msa_search` | Search memory bank | Before answering questions about prior context |
| `msa_store` | Store new information | After responses containing retainable info |
| `msa_multihop` | Multi-hop reasoning | Complex questions requiring evidence chaining |

---

## Building Your Own Integration

All integrations follow the same pattern:

1. **Initialise**: Load a `TieredMemoryBank` with your encoded corpus
2. **Search**: Use `route_query()` to find relevant documents
3. **Inject**: Format retrieved context for your agent's prompt format
4. **Store**: Queue new information for batch re-encoding

```python
from msa_memory.config import MSAConfig
from msa_memory.parallel import TieredMemoryBank
from msa_memory.generate import encode_query_text
from msa_memory.route import route_query

# 1. Initialise
config = MSAConfig()
bank = TieredMemoryBank(config)
bank.load("./msa_memory_bank/")

# 2. Search
query_hidden = encode_query_text("your query", model, tokenizer)
K_ctx, V_ctx, doc_ids = route_query(
    query_hidden, bank.kr_cache, bank.k_cache, bank.v_cache,
    config, router_q_proj
)

# 3. Inject: format for your agent
context = "\n".join(bank.doc_index.get(str(i), "") for i in doc_ids)
```
