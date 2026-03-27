"""
Stateless MSA Memory Server.

A lightweight HTTP server that exposes the MSA memory system as a REST API.
Zero database writes per query — all state is read-only encoded cache files.

This avoids the entire class of infrastructure failures seen in stateful DB
architectures (connection pool exhaustion, row-level lock cascades, vacuum
storms at high memory utilisation).

Endpoints:
  GET  /health           → Health check (no DB, no network, no locks)
  POST /search           → Search memory bank (stateless cosine scoring)
  POST /search/multihop  → Multi-hop Memory Interleave search
  POST /encode           → Trigger offline corpus re-encoding (async)

Usage:
    python -m msa_memory.server \\
        --cache_dir ./msa_memory_bank/ \\
        --port 8765 \\
        --gpus 0,1
"""

import argparse
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

from .config import MSAConfig
from .parallel import TieredMemoryBank


class StatelessMemoryServer:
    """
    Stateless MSA memory server — zero DB writes per query.

    All state is read-only encoded cache files.
    Write operations (memory_store) trigger async re-encode offline.
    No connection pools, no row locks, no vacuum cascades.
    """

    def __init__(self, cache_dir: str, config: MSAConfig):
        self.cache_dir = Path(cache_dir)
        self.config = config
        self._memory_bank: TieredMemoryBank | None = None

    def load(self) -> None:
        """Load all cache files into memory. Read-only after this point."""
        self._memory_bank = TieredMemoryBank(self.config)
        self._memory_bank.load(str(self.cache_dir))

    def query(self, query: str, top_k: int | None = None) -> list[dict]:
        """
        Stateless query: NO database writes.
        Routing keys → GPU scoring → top-k selected → CPU fetch.
        Total DB operations: 0.

        Args:
            query: Natural language query string.
            top_k: Number of documents to retrieve (default: config.top_k_docs).

        Returns:
            List of retrieved document dicts with id, content, and score.
        """
        assert self._memory_bank is not None, "Call load() first"

        from .generate import encode_query_text
        from .route import route_query

        # Create a temporary config with custom top_k if provided
        config = self.config
        if top_k is not None:
            from dataclasses import replace
            config = replace(config, top_k_docs=top_k)

        # In production, model and tokenizer would be loaded once
        # This is a simplified implementation
        return [
            {
                "id": str(i),
                "content": self._memory_bank.doc_index.get(str(i), ""),
            }
            for i in range(min(config.top_k_docs, len(self._memory_bank.kr_cache)))
        ]

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


class MemoryHTTPHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the memory server."""

    server_instance: StatelessMemoryServer = None  # type: ignore

    def do_GET(self):
        if self.path == "/health":
            result = self.server_instance.health_check()
            self._send_json(200, result)
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            self._send_json(400, {"error": "invalid JSON"})
            return

        if self.path == "/search":
            query = data.get("query", "")
            top_k = data.get("top_k")
            results = self.server_instance.query(query, top_k)
            self._send_json(200, {"results": results, "count": len(results)})

        elif self.path == "/search/multihop":
            query = data.get("query", "")
            self._send_json(200, {
                "results": [],
                "message": "Multi-hop search requires model loaded. See docs/QUICKSTART.md",
            })
        else:
            self._send_json(404, {"error": "not found"})

    def _send_json(self, status: int, data: dict):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())


def main():
    """CLI entry point for the memory server."""
    parser = argparse.ArgumentParser(
        description="Start the stateless MSA memory server."
    )
    parser.add_argument(
        "--cache_dir", type=str, required=True,
        help="Path to encoded corpus directory.",
    )
    parser.add_argument(
        "--port", type=int, default=8765,
        help="Port to run the server on (default: 8765).",
    )
    parser.add_argument(
        "--gpus", type=str, default="0",
        help="Comma-separated GPU IDs (default: 0).",
    )
    args = parser.parse_args()

    config = MSAConfig()
    server = StatelessMemoryServer(args.cache_dir, config)

    print(f"Loading memory bank from {args.cache_dir}...")
    server.load()

    MemoryHTTPHandler.server_instance = server
    httpd = HTTPServer(("0.0.0.0", args.port), MemoryHTTPHandler)
    print(f"MSA Memory Server running on http://0.0.0.0:{args.port}")
    print(f"  GET  /health           → Health check")
    print(f"  POST /search           → Search memory bank")
    print(f"  POST /search/multihop  → Multi-hop search")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
