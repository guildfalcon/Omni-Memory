"""
Patch: Infrastructure Reliability.

Supermemory failure: March 6, 2026 outage — a single UPDATE apikey query
at 95% DB RAM utilisation caused a cascade.

MSA avoids this entire failure class because:
  1. Memory bank is static files (K̄, V̄, K̄ᴿ) — no live DB writes per request
  2. Routing is a stateless cosine scoring pass — no row-level locking
  3. No rate-limit counters in the hot path

This patch implements a stateless memory server following these principles.
"""

from pathlib import Path

from msa_memory.config import MSAConfig
from msa_memory.parallel import TieredMemoryBank
from patches.fix_precision_ceiling import PrecisionPatch


class StatelessMemoryServer:
    """
    Stateless MSA memory server — zero DB writes per query.

    All state is read-only encoded cache files.
    Write operations (memory_store) trigger async re-encode offline.
    No connection pools, no row locks, no vacuum cascades.

    Usage:
        server = StatelessMemoryServer("./msa_memory_bank/", config)
        server.load()
        results = server.query("What are the user's preferences?")
        health = server.health_check()
    """

    def __init__(self, cache_dir: str, config: MSAConfig):
        self.cache_dir = Path(cache_dir)
        self.config = config
        self._memory_bank: TieredMemoryBank | None = None

    def load(self) -> None:
        """Load all cache files into memory. Read-only after this point."""
        self._memory_bank = TieredMemoryBank(self.config)
        self._memory_bank.load(str(self.cache_dir))

    def query(self, query: str) -> list[dict]:
        """
        Stateless query: NO database writes.
        Routing keys → GPU scoring → top-k selected → CPU fetch.
        Total DB operations: 0.

        Args:
            query: Natural language search query.

        Returns:
            List of retrieved document dicts.
        """
        assert self._memory_bank is not None, "Call load() first"
        return PrecisionPatch(self._memory_bank, self.config).search(query)

    def health_check(self) -> dict:
        """
        Health check that cannot cascade: just checks file existence.
        No DB connection, no network call, no lock acquisition.
        """
        return {
            "status": "ok",
            "routing_keys_loaded": (
                len(self._memory_bank.kr_cache) if self._memory_bank else 0
            ),
            "cache_dir_exists": self.cache_dir.exists(),
            "last_encode": (
                (self.cache_dir / "routing_keys.pt").stat().st_mtime
                if (self.cache_dir / "routing_keys.pt").exists()
                else None
            ),
        }
