"""
Stage 1 — Offline Corpus Encoding.

This module handles the one-time offline encoding of your document corpus into
compressed KV cache + routing keys. Run this whenever your corpus changes.

The encoding produces three compressed matrices per document:
  - K̄  (compressed standard keys)   → stored in CPU DRAM
  - V̄  (compressed standard values)  → stored in CPU DRAM
  - K̄ᴿ (compressed routing keys)    → stored in GPU VRAM

Cost: O(L·G) amortised across all future queries.

Usage:
    python -m msa_memory.encode \\
        --corpus_dir ./my_documents/ \\
        --output_dir ./msa_memory_bank/ \\
        --chunk_size 64 \\
        --backbone Qwen/Qwen3-4B-Instruct
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .config import MSAConfig


def encode_corpus(
    documents: list[str],
    config: MSAConfig,
    output_dir: str,
    model,
    tokenizer,
) -> None:
    """
    One-time offline encoding. Run this whenever your corpus changes.
    Outputs three compressed matrices per document: K̄, V̄, K̄ᴿ

    This is Stage 1 — amortised cost across all future queries.
    Cost: O(L·G) once, not per query.

    Args:
        documents: List of document text strings.
        config: MSAConfig instance with encoding parameters.
        output_dir: Directory to write encoded cache files.
        model: HuggingFace model with k_proj, v_proj, kr_proj heads.
        tokenizer: Corresponding tokenizer.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Determine which layers to apply MSA routing to (latter half)
    num_layers = model.config.num_hidden_layers
    msa_start_layer = num_layers // 2  # e.g., layer 14 of 28

    k_cache = []   # compressed standard keys
    v_cache = []   # compressed standard values
    kr_cache = []  # compressed routing keys

    model.eval()
    with torch.no_grad():
        for doc_id, doc_text in enumerate(documents):
            tokens = tokenizer(
                doc_text,
                return_tensors="pt",
                truncation=False,
                padding=False,
            ).input_ids.to(model.device)

            # Forward pass — collect hidden states for MSA layers only
            outputs = model(
                tokens,
                output_hidden_states=True,
                return_dict=True,
            )

            # Extract KV from MSA-applicable layers (latter half)
            # NOTE: Exact API depends on backbone; adapt as needed
            hidden = outputs.hidden_states[msa_start_layer]  # [1, seq_len, d_model]

            # Project to K, V, Kᴿ using backbone weight matrices
            # (In practice, add Router K Projector as a new nn.Linear)
            K = model.k_proj(hidden)    # [1, seq_len, num_heads * head_dim]
            V = model.v_proj(hidden)
            KR = model.kr_proj(hidden)  # NEW router projection head

            # Chunk-wise mean pooling (compress by factor P)
            K_bar = chunk_mean_pool(K, config.chunk_size)   # [1, n_chunks, d]
            V_bar = chunk_mean_pool(V, config.chunk_size)
            KR_bar = chunk_mean_pool(KR, config.chunk_size)

            k_cache.append(K_bar.cpu())
            v_cache.append(V_bar.cpu())
            kr_cache.append(KR_bar.cpu())

            if doc_id % 500 == 0:
                print(f"Encoded {doc_id}/{len(documents)} documents")

    # Save tiered: routing keys → GPU-ready format, content KVs → CPU DRAM
    torch.save(kr_cache, output_path / "routing_keys.pt")   # goes to VRAM
    torch.save(k_cache, output_path / "content_k.pt")       # stays in DRAM
    torch.save(v_cache, output_path / "content_v.pt")       # stays in DRAM

    # Save document metadata (ID → original text mapping for Memory Interleave)
    with open(output_path / "doc_index.json", "w") as f:
        json.dump({str(i): doc[:500] for i, doc in enumerate(documents)}, f)

    print(f"Encoding complete. {len(documents)} documents → {output_dir}")


def chunk_mean_pool(tensor: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """
    φ(·): compress a [batch, seq_len, d] tensor via chunk-wise mean pooling.

    This is the core compression function that reduces KV cache size by a factor
    of `chunk_size` (P in the paper). Each chunk of P consecutive token vectors
    is replaced by their mean, producing a single representative vector.

    Args:
        tensor: Input tensor of shape [batch, seq_len, d_model].
        chunk_size: Number of tokens per chunk (P parameter).

    Returns:
        Compressed tensor of shape [batch, n_chunks, d_model]
        where n_chunks = ceil(seq_len / chunk_size).
    """
    B, L, D = tensor.shape
    # Pad to multiple of chunk_size
    pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        tensor = F.pad(tensor, (0, 0, 0, pad_len))
    # Reshape and average
    n_chunks = (L + pad_len) // chunk_size
    return tensor.view(B, n_chunks, chunk_size, D).mean(dim=2)  # [B, n_chunks, D]


def _load_documents_from_dir(corpus_dir: str) -> list[str]:
    """Load all text files from a directory as documents."""
    corpus_path = Path(corpus_dir)
    documents = []
    for ext in ["*.txt", "*.md", "*.py", "*.json", "*.html"]:
        for file_path in sorted(corpus_path.rglob(ext)):
            try:
                text = file_path.read_text(encoding="utf-8")
                if text.strip():
                    documents.append(text)
            except (UnicodeDecodeError, PermissionError):
                print(f"Skipping unreadable file: {file_path}")
    return documents


def main():
    """CLI entry point for corpus encoding."""
    parser = argparse.ArgumentParser(
        description="Encode a document corpus into MSA memory bank format."
    )
    parser.add_argument(
        "--corpus_dir", type=str, required=True,
        help="Directory containing documents to encode."
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory to write encoded cache files."
    )
    parser.add_argument(
        "--chunk_size", type=int, default=64,
        help="Tokens per compression chunk (default: 64)."
    )
    parser.add_argument(
        "--backbone", type=str, default="Qwen/Qwen3-4B-Instruct",
        help="HuggingFace model ID for the backbone."
    )
    args = parser.parse_args()

    config = MSAConfig(
        backbone_model=args.backbone,
        chunk_size=args.chunk_size,
    )

    print(f"Loading backbone model: {config.backbone_model}")
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.backbone_model)
    model = AutoModel.from_pretrained(
        config.backbone_model, output_hidden_states=True
    )

    print(f"Loading documents from: {args.corpus_dir}")
    documents = _load_documents_from_dir(args.corpus_dir)
    print(f"Found {len(documents)} documents")

    encode_corpus(documents, config, args.output_dir, model, tokenizer)


if __name__ == "__main__":
    main()
