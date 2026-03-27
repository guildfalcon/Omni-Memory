# Contributing to Omni-Memory

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to Omni-Memory.

---

## Getting Started

### 1. Fork & Clone

```bash
git clone https://github.com/YOUR_USERNAME/omni-memory.git
cd omni-memory
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dev dependencies
pip install -r requirements-dev.txt

# Install package in editable mode
pip install -e .
```

### 3. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

---

## Development Workflow

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black msa_memory/ integrations/ patches/ tests/
```

### Linting

```bash
ruff check msa_memory/ integrations/ patches/
```

### Type Checking

```bash
mypy msa_memory/
```

---

## Contribution Guidelines

### Code Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) conventions
- Use type hints for all function signatures
- Write docstrings for all public functions and classes
- Keep lines under 100 characters (configured in `pyproject.toml`)

### Commit Messages

Use clear, descriptive commit messages:

```
feat: add incremental encoding for live buffer
fix: correct chunk padding in chunk_mean_pool
docs: update training guide with 64K extension details
test: add unit tests for routing score calculation
```

### Pull Requests

1. **One feature per PR** — keep changes focused
2. **Include tests** — new features should have corresponding tests
3. **Update documentation** — if your change affects user-facing behaviour
4. **Run the full test suite** before submitting

### What We're Looking For

- **Bug fixes** — always welcome
- **Performance improvements** — especially for routing and encoding
- **New integrations** — plugins for additional agent frameworks
- **Documentation improvements** — clearer explanations, more examples
- **Incremental encoding** — solving the static corpus limitation
- **Larger backbone support** — training on 8B+ models

---

## Architecture Overview

Before contributing, read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) to understand:

- The three-stage memory pipeline (encode → route → generate)
- Why MSA routing only applies to the latter half of layers
- How doc-wise RoPE enables scale generalisation
- The tiered storage strategy for 100M-token inference

---

## Reporting Issues

Use the [GitHub issue tracker](https://github.com/yourusername/omni-memory/issues) with the provided templates:

- **Bug Report** — for things that aren't working as expected
- **Feature Request** — for new capabilities or improvements

---

## License

By contributing to Omni-Memory, you agree that your contributions will be licensed under the [MIT License](LICENSE).
