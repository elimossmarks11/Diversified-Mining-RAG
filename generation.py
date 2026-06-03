"""Generation script for the RAG pipeline.

Retrieves context (or loads pre-computed retrieval JSON) and generates an
answer using Qwen2.5-0.5B-Instruct.  Can be run standalone for debugging;
the primary CLI entry point is pipeline.py.

Usage:
    python generation.py "What are Anglo American's GHG targets?"
    python generation.py "query" --retrieval-json results.json --output answer.json
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

from utils import RetrievalConfig, generate_answer

logger = logging.getLogger(__name__)


def main() -> None:
    """Parse a query from argv and run generation.

    Reads the first positional argument as the query string.
    Optionally accepts ``--retrieval-json <path>`` to skip live retrieval
    and ``--output <path>`` to write the result to a file.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if len(sys.argv) < 2:
        logger.error(
            "Usage: python generation.py <query> "
            "[--retrieval-json path.json] [--output path.json]"
        )
        sys.exit(1)

    query = sys.argv[1]

    # Optional --retrieval-json flag
    retrieval_result: dict | None = None
    if "--retrieval-json" in sys.argv:
        idx = sys.argv.index("--retrieval-json")
        if idx + 1 < len(sys.argv):
            rpath = Path(sys.argv[idx + 1])
            logger.info("Loading retrieval results from %s", rpath)
            retrieval_result = json.loads(rpath.read_text(encoding="utf-8"))

    # Optional --output flag
    output: str | None = None
    if "--output" in sys.argv:
        idx = sys.argv.index("--output")
        if idx + 1 < len(sys.argv):
            output = sys.argv[idx + 1]

    config = RetrievalConfig()
    logger.info("Generating answer for: %r", query)
    result = generate_answer(query, config=config, retrieval_result=retrieval_result)
    payload = json.dumps(result, indent=2, ensure_ascii=False)

    if output:
        Path(output).write_text(payload, encoding="utf-8")
        logger.info("Result written to %s", output)
    else:
        print(payload)  # noqa: T201 — stdout output for piping


if __name__ == "__main__":
    main()