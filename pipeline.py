"""pipeline.py — CLI orchestrator for the TPI RAG pipeline.

Commands:
    python pipeline.py chunk      – run chunking for all PDFs
    python pipeline.py embed      – embed chunks into ChromaDB
    python pipeline.py retrieve   – retrieve top-k chunks for a query
    python pipeline.py generate   – retrieve + generate an answer
    python pipeline.py run        – full pipeline: chunk → embed → retrieve → generate
    python pipeline.py serve      – launch FastAPI server
"""

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

# Windows DLL workaround (see utils.py docstring)
if os.name == "nt":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    import torch  # noqa: E402, F401

import click

from utils import RetrievalConfig, generate_answer, retrieve_chunks

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------

INTERIM_DIR = Path("data/interim")
RAW_DIR = Path("data/raw")


def get_company_dir(company: str) -> Path:
    """Return the raw data directory for a company.

    Args:
        company: Company identifier (e.g. ``antam``, ``anglo``).

    Returns:
        Path to ``data/raw/{company}``.
    """
    return RAW_DIR / company


def get_interim_dir(company: str) -> Path:
    """Return (and create) the interim output directory for a company.

    Args:
        company: Company identifier.

    Returns:
        Path to ``data/interim/{company}``.
    """
    path = INTERIM_DIR / company
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_chunks_path(company: str) -> Path:
    """Return the parquet path for a company's chunked output.

    Args:
        company: Company identifier.

    Returns:
        Path to ``data/interim/{company}/chunks.parquet``.
    """
    return get_interim_dir(company) / "chunks.parquet"


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


@click.group()
def cli() -> None:
    """RAG pipeline for TPI Carbon Performance extraction."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# ---------------------------------------------------------------------------
# Individual stage commands
# ---------------------------------------------------------------------------


@cli.command()
@click.option(
    "--company",
    default=None,
    help="Process a single company (antam or anglo). Default: all.",
)
def chunk(company: str | None) -> None:
    """Run chunking for PDFs (delegates to chunking.py)."""
    cmd = [sys.executable, "chunking.py", "chunk"]
    if company:
        cmd.extend(["--company", company])
    click.echo(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        logger.error("Chunking failed")
        raise
    click.echo("Chunking complete.")


@cli.command()
def embed() -> None:
    """Embed chunks into ChromaDB (delegates to vector_store.py)."""
    cmd = [sys.executable, "vector_store.py"]
    click.echo(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        logger.error("Embedding failed")
        raise
    click.echo("Embedding complete.")


@cli.command()
@click.argument("query")
@click.option(
    "--final-k",
    default=RetrievalConfig().final_k,
    type=int,
    help="Number of final results.",
)
@click.option(
    "--output", "-o", default=None, type=click.Path(), help="Write JSON to file."
)
def retrieve(query: str, final_k: int, output: str | None) -> None:
    """Retrieve top-k chunks for QUERY."""
    config = RetrievalConfig(final_k=final_k)
    click.echo(f"Retrieving top-{final_k} chunks for: {query!r}")
    result = retrieve_chunks(query, config)
    payload = json.dumps(result, indent=2, ensure_ascii=False)

    if output:
        Path(output).write_text(payload, encoding="utf-8")
        click.echo(f"Results written to {output}")
    else:
        click.echo(payload)


@cli.command()
@click.argument("query")
@click.option(
    "--final-k",
    default=RetrievalConfig().final_k,
    type=int,
    help="Number of chunks to use.",
)
@click.option(
    "--max-tokens",
    default=RetrievalConfig().max_new_tokens,
    type=int,
    help="Max new tokens.",
)
@click.option(
    "--output", "-o", default=None, type=click.Path(), help="Write JSON to file."
)
def generate(query: str, final_k: int, max_tokens: int, output: str | None) -> None:
    """Retrieve context and generate an answer for QUERY."""
    config = RetrievalConfig(final_k=final_k, max_new_tokens=max_tokens)
    click.echo(f"Generating answer for: {query!r}")
    result = generate_answer(query, config=config)
    payload = json.dumps(result, indent=2, ensure_ascii=False)

    if output:
        Path(output).write_text(payload, encoding="utf-8")
        click.echo(f"Result written to {output}")
    else:
        click.echo(payload)


@cli.command()
@click.argument("query")
@click.option(
    "--company",
    default=None,
    help="Process a single company (antam or anglo). Default: all.",
)
@click.option(
    "--skip-chunk", is_flag=True, help="Skip chunking (use existing parquets)."
)
@click.option(
    "--skip-embed", is_flag=True, help="Skip embedding (use existing ChromaDB)."
)
@click.option("--final-k", default=RetrievalConfig().final_k, type=int)
@click.option("--max-tokens", default=RetrievalConfig().max_new_tokens, type=int)
@click.option("--output", "-o", default=None, type=click.Path())
def run(
    query: str,
    company: str | None,
    skip_chunk: bool,
    skip_embed: bool,
    final_k: int,
    max_tokens: int,
    output: str | None,
) -> None:
    """Full pipeline: chunk → embed → retrieve → generate for QUERY."""
    if not skip_chunk:
        click.echo("=== Stage 1: Chunking ===")
        cmd = [sys.executable, "chunking.py", "chunk"]
        if company:
            cmd.extend(["--company", company])
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            logger.error("Chunking stage failed")
            raise

    if not skip_embed:
        click.echo("=== Stage 2: Embedding ===")
        try:
            subprocess.run([sys.executable, "vector_store.py"], check=True)
        except subprocess.CalledProcessError:
            logger.error("Embedding stage failed")
            raise

    click.echo("=== Stage 3: Retrieval ===")
    config = RetrievalConfig(final_k=final_k, max_new_tokens=max_tokens)
    retrieval_result = retrieve_chunks(query, config)
    click.echo(f"Retrieved {len(retrieval_result['ids'])} chunks.")

    click.echo("=== Stage 4: Generation ===")
    result = generate_answer(query, config=config, retrieval_result=retrieval_result)

    payload = json.dumps(result, indent=2, ensure_ascii=False)
    if output:
        Path(output).write_text(payload, encoding="utf-8")
        click.echo(f"Result written to {output}")
    else:
        click.echo(payload)

    click.echo("=== Pipeline complete ===")


@cli.command()
@click.option("--host", default="127.0.0.1", help="Bind address.")
@click.option("--port", default=8000, type=int, help="Port number.")
def serve(host: str, port: int) -> None:
    """Launch the FastAPI server (delegates to uvicorn)."""
    # Lazy import: uvicorn is only needed for the serve command
    import uvicorn

    click.echo(f"Starting server at http://{host}:{port}")
    uvicorn.run("app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    cli()