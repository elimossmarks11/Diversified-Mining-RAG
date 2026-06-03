"""vector_store.py — Embed chunks and store in a persistent ChromaDB collection.

Reads .parquet chunk files produced by chunking.py, embeds with
all-MiniLM-L6-v2, and upserts into a single ChromaDB collection.

Usage:
    python vector_store.py                    # embed all companies
    python vector_store.py --company anglo    # embed one company
    python vector_store.py --force            # overwrite existing collection
"""

import logging
import os
from pathlib import Path

# Workaround: conda numpy (MKL) and pip torch ship conflicting OpenMP
# runtimes on Windows.  Allow both to coexist for CPU-only inference.
if os.name == "nt":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch  # noqa: E402 — must load before pandas/numpy on Windows (fbgemm.dll)

import click
import pandas as pd

from utils import embed_texts, get_chroma_client, load_embedding_model, upsert_to_collection

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Embedding model — chosen after MiniLM vs harrier comparison
# (see testing/embedding_tests/embedding_comparison.ipynb)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Root paths — resolved from this file's location so CWD doesn't matter
ROOT = Path(__file__).resolve().parent
INTERIM_DIR = ROOT / "data" / "interim"

# ChromaDB persistence directory — intermediate output, re-creatable from parquet
CHROMA_DIR = INTERIM_DIR / "chromadb"

# Single collection for all chunks; use metadata filters for company/source queries
COLLECTION_NAME = "tpi_chunks"

# Batch sizes for encoding and ChromaDB upsert
EMBEDDING_BATCH_SIZE = 256
CHROMA_UPSERT_BATCH_SIZE = 256

# Company labels matching chunking.py's COMPANY_FOLDERS keys
COMPANY_LABELS: list[str] = ["antam", "anglo"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_chunks_path(label: str) -> Path:
    """Return path to the combined chunks parquet for a company."""
    return INTERIM_DIR / label / "chunks.parquet"


def _build_metadata(row: pd.Series, company: str) -> dict[str, str | int | float | bool]:
    """Build a ChromaDB-compatible metadata dict from a parquet row.

    ChromaDB metadata values must be scalars (str, int, float, bool).
    List columns are serialised to their string representation.

    Args:
        row: A single row from the chunks DataFrame.
        company: Company label (e.g. 'anglo', 'antam').

    Returns:
        Dict of scalar metadata values for ChromaDB storage.
    """
    meta: dict[str, str | int | float | bool] = {"company": company}

    for str_col in ("source", "strategy"):
        val = row.get(str_col)
        if pd.notna(val):
            meta[str_col] = str(val)

    if pd.notna(row.get("page_num")):
        meta["page_num"] = int(row["page_num"])

    if row.get("is_table") is True:
        meta["is_table"] = True

    # ChromaDB metadata values must be scalars — serialise lists to strings.
    # Parquet round-trips may return numpy arrays/int64 instead of Python lists,
    # so convert elements to plain Python ints before str() serialisation.
    for list_col in ("pages", "element_types"):
        val = row.get(list_col)
        if val is not None and not isinstance(val, str) and hasattr(val, "__iter__"):
            meta[list_col] = str([int(x) if isinstance(x, (int, float)) else str(x) for x in val])

    return meta


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option(
    "--company",
    type=click.Choice(COMPANY_LABELS),
    default=None,
    help="Embed a single company. Omit to embed all.",
)
@click.option("--force", is_flag=True, help="Delete and recreate the collection.")
def embed(company: str | None, force: bool) -> None:
    """Embed parquet chunks into a persistent ChromaDB collection.

    Reads company-level chunks.parquet files from data/interim/{company}/,
    embeds with all-MiniLM-L6-v2, and upserts into the 'tpi_chunks'
    collection at data/interim/chromadb/.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    labels = [company] if company else COMPANY_LABELS

    # -- Load chunks from parquet ----------------------------------------
    all_dfs: list[pd.DataFrame] = []
    for label in labels:
        path = _get_chunks_path(label)
        if not path.exists():
            click.echo(f"WARNING: No chunks at {path} — run chunking.py chunk first.")
            continue
        df = pd.read_parquet(path)
        df["company"] = label
        all_dfs.append(df)
        click.echo(f"Loaded {len(df)} chunks from {path}")

    if not all_dfs:
        raise click.ClickException("No chunk files found. Run 'python chunking.py chunk' first.")

    combined = pd.concat(all_dfs, ignore_index=True)
    click.echo(f"\nTotal: {len(combined)} chunks to embed")

    # -- Check for existing collection (before expensive embedding) ------
    client = get_chroma_client(CHROMA_DIR)

    existing_names = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing_names:
        if not force:
            click.echo(
                f"Collection '{COLLECTION_NAME}' already exists. Use --force to overwrite."
            )
            return
        client.delete_collection(COLLECTION_NAME)
        click.echo(f"Deleted existing collection '{COLLECTION_NAME}'")

    # -- Embed -----------------------------------------------------------
    click.echo(f"\nLoading model: {EMBEDDING_MODEL_NAME}")
    model = load_embedding_model(EMBEDDING_MODEL_NAME)
    click.echo(f"Model loaded (dim={model.get_sentence_embedding_dimension()})")

    texts = combined["text"].tolist()
    metadatas = [_build_metadata(row, row["company"]) for _, row in combined.iterrows()]

    # Chunk IDs from chunking (e.g. elem_0000) are only unique per-PDF.
    # Prefix with the source label to make them globally unique.
    combined["global_id"] = combined["source"].fillna("") + "_" + combined["id"]
    ids = combined["global_id"].tolist()

    # Sanity check: abort early if IDs still collide
    n_dupes = combined["global_id"].duplicated().sum()
    if n_dupes > 0:
        examples = combined.loc[combined["global_id"].duplicated(keep=False), "global_id"].unique()[:5]
        raise click.ClickException(
            f"Duplicate IDs detected after prefixing ({n_dupes} dupes). "
            f"Examples: {list(examples)}"
        )

    click.echo(f"Embedding {len(texts)} chunks...")
    embeddings = embed_texts(model, texts, batch_size=EMBEDDING_BATCH_SIZE)

    # -- Create collection and upsert (after embedding succeeds) ---------
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    upsert_to_collection(
        collection,
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
        batch_size=CHROMA_UPSERT_BATCH_SIZE,
    )

    click.echo(f"\n{len(ids)} chunks embedded into '{COLLECTION_NAME}' at {CHROMA_DIR}")


if __name__ == "__main__":
    embed()