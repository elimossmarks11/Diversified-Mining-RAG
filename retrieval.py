"""Retrieval script for the RAG pipeline.

Performs hybrid retrieval (dense + BM25 → RRF → cross-encoder reranking)
against the ChromaDB vector store.  Can be run standalone for debugging;
the primary CLI entry point is pipeline.py.

Usage:
    python retrieval.py "What are Anglo American's GHG targets?"
"""

import json
import logging
import os
import sys
from pathlib import Path

# Windows DLL workaround (see utils.py docstring)
if os.name == "nt":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    import torch  # noqa: E402, F401

from utils import RetrievalConfig, retrieve_chunks

logger = logging.getLogger(__name__)


def main() -> None:
    """Parse a query from argv and run hybrid retrieval.

    Reads the first positional argument as the query string,
    runs ``retrieve_chunks``, and writes the result as JSON to stdout.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if len(sys.argv) < 2:
        logger.error("Usage: python retrieval.py <query> [--output path.json]")
        sys.exit(1)

    query = sys.argv[1]

    # Optional --output flag
    output: str | None = None
    if "--output" in sys.argv:
        idx = sys.argv.index("--output")
        if idx + 1 < len(sys.argv):
            output = sys.argv[idx + 1]

    config = RetrievalConfig()
    logger.info("Retrieving top-%d chunks for: %r", config.final_k, query)
    result = retrieve_chunks(query, config)
    payload = json.dumps(result, indent=2, ensure_ascii=False)

    if output:
        Path(output).write_text(payload, encoding="utf-8")
        logger.info("Results written to %s", output)
    else:
        print(payload)  # noqa: T201 — stdout output for piping


if __name__ == "__main__":
    main()