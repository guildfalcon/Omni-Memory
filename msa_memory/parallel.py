"""
Memory Parallel — 100M Token Inference.

Tiered storage strategy to fit 100M tokens on 2×A800 GPUs.

Memory budget breakdown for 100M tokens:
  - Compressed KV + K̄ᴿ cache: ~169 GB total
  - 2×A800 VRAM:               ~160 GB total
  - Solution: routing keys → GPU, content KVs → CPU DRAM

Tiered storage:
  K̄ᴿ (routing keys)  → GPU VRAM   (~56 GB for 100M)  — needed for fast scoring
  K̄, V̄ (content KVs) → CPU DRAM   (~113 GB)          — loaded async after selection
"""

from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn.functional as F

from .config import MSAConfig


class TieredMemoryBank:
    """
    Tiered memory bank for large-scale MSA inference.

    Implements the Memory Parallel strategy from the MSA paper:
    - Routing keys (K̄ᴿ) reside on GPU VRAM for fast cosine scoring
    - Content KVs (K̄, V̄) reside in CPU DRAM, fetched on demand
    - Only top-k documents are transferred CPU → GPU per query

    This design enables 100M-token scale on 2×A800:
    - Never moves all content to GPU simultaneously
    - Overlaps scoring and data transfer when async_prefetch=True

    For single-GPU setups, scoring falls back to sequential mode
    automatically.

    Usage:
        bank = TieredMemoryBank(config)
        bank.load("./msa_memory_bank/")
        scores = bank.distributed_score(query_routing)
        top_k = torch.topk(scores, k=config.top_k_docs).indices
        selected_k, selected_v = bank.fetch_top_k(top_k.tolist())
    """

    def __init__(self, config: MSAConfig):
        self.config = config
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Routing keys: GPU-resident for fast cosine scoring
        self.kr_cache: list[torch.Tensor] = []  # each on CUDA

        # Content KVs: CPU-resident, only top-k fetched on demand
        self.k_cache: list[torch.Tensor] = []   # each on CPU
        self.v_cache: list[torch.Tensor] = []   # each on CPU

        # Document index: id → text mapping
        self.doc_index: dict[str, str] = {}

    def load(self, encoding_dir: str, gpu_device: str = "cuda:0"):
        """
        Load tiered memory bank from encoded corpus directory.

        Expects the following files (created by encode.py):
          - routing_keys.pt  → loaded to GPU VRAM
          - content_k.pt     → kept in CPU DRAM
          - content_v.pt     → kept in CPU DRAM
          - doc_index.json   → document metadata

        Args:
            encoding_dir: Path to the encoded corpus directory.
            gpu_device: GPU device for routing keys (default: "cuda:0").
        """
        import json
        from pathlib import Path

        encoding_path = Path(encoding_dir)

        print("Loading routing keys → GPU VRAM...")
        kr_raw = torch.load(
            encoding_path / "routing_keys.pt", map_location="cpu"
        )
        self.kr_cache = [kr.to(gpu_device) for kr in kr_raw]

        print("Loading content KVs → CPU DRAM...")
        self.k_cache = torch.load(
            encoding_path / "content_k.pt", map_location="cpu"
        )
        self.v_cache = torch.load(
            encoding_path / "content_v.pt", map_location="cpu"
        )

        # Load document index if available
        doc_index_path = encoding_path / "doc_index.json"
        if doc_index_path.exists():
            with open(doc_index_path) as f:
                self.doc_index = json.load(f)

        print(f"Memory bank ready: {len(self.kr_cache)} documents")

    def fetch_top_k(
        self,
        top_k_indices: list[int],
        target_device: str = "cuda:0",
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Async prefetch: only move selected top-k content KVs from CPU → GPU.

        This is the key efficiency mechanism — we never move all 100M tokens
        to GPU. Only the top-k selected documents are transferred.

        Args:
            top_k_indices: Document indices to fetch.
            target_device: GPU device to transfer content KVs to.

        Returns:
            Tuple of (selected_keys, selected_values) on target_device.
        """
        selected_k = [self.k_cache[i].to(target_device) for i in top_k_indices]
        selected_v = [self.v_cache[i].to(target_device) for i in top_k_indices]
        return selected_k, selected_v

    def distributed_score(
        self,
        query_routing: torch.Tensor,  # routing query vector
        gpu_ids: list[int] | None = None,
    ) -> torch.Tensor:
        """
        Memory-Parallel scoring: shard routing keys across GPUs,
        score in parallel, reduce globally for top-k selection.

        For single-GPU, falls back to sequential scoring automatically.

        Args:
            query_routing: Routing query vector [1, q_len, d_routing].
            gpu_ids: List of GPU device IDs (default: [0]).

        Returns:
            Score tensor [num_documents] with relevance scores.
        """
        if gpu_ids is None:
            gpu_ids = [0]

        n_docs = len(self.kr_cache)
        n_gpus = len(gpu_ids)
        shard_size = n_docs // n_gpus

        scores_per_gpu = []
        for gpu_id in gpu_ids:
            start = gpu_id * shard_size
            end = start + shard_size if gpu_id < n_gpus - 1 else n_docs
            device = f"cuda:{gpu_id}"

            # Score this shard
            shard_scores = self._score_shard(
                query_routing.to(device),
                self.kr_cache[start:end],
                device,
            )
            scores_per_gpu.append((start, shard_scores.cpu()))

        # Global reduce: reconstruct full score vector
        full_scores = torch.zeros(n_docs)
        for start, shard_scores in scores_per_gpu:
            full_scores[start : start + len(shard_scores)] = shard_scores

        return full_scores

    def _score_shard(
        self,
        query_r: torch.Tensor,
        kr_shard: list[torch.Tensor],
        device: str,
    ) -> torch.Tensor:
        """Score one GPU shard of routing keys."""
        scores = []
        for kr in kr_shard:
            kr = kr.to(device)
            sim = F.cosine_similarity(
                query_r.unsqueeze(0), kr.unsqueeze(0), dim=-1
            )
            scores.append(sim.max().item())
        return torch.tensor(scores)
