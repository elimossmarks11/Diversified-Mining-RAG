"""Three-stage chunking pipeline: classify → extract-tables → chunk.

Stage 1 (classify):       pdfplumber numeric density → table_pages.json  [rag env]
Stage 2 (extract-tables): Docling on table pages only → docling.json     [docling_test env]
Stage 3 (chunk):          element_type prose + Docling table → parquet   [rag env]

Automated overnight via run_chunking.bat (switches conda envs automatically).
"""

import json
import logging
import os
import re
from pathlib import Path

# Workaround: conda numpy (MKL) and pip torch ship conflicting OpenMP
# runtimes on Windows. Allow both to coexist for CPU-only inference.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import click

# All heavy imports are lazy — inside their respective commands — so that
# extract-tables can run in the docling_test env without pdfplumber/gensim/etc.

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Root paths — all relative to the repository root
ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
INTERIM_DIR = ROOT / "data" / "interim"

# Company label → subfolder under data/raw/
COMPANY_FOLDERS: dict[str, str] = {
    "antam": "antam",
    "anglo": "anglo",
}

# Docling batching limit — caps peak RAM to avoid std::bad_alloc on large PDFs
DOCLING_BATCH_SIZE = 30

# Char limit for element_type chunking (matches evaluation hyperparameter)
CHAR_LIMIT = 1000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_pdf_files(label: str) -> list[Path]:
    """Return all .pdf files in the raw folder for a company, sorted."""
    folder = RAW_DIR / COMPANY_FOLDERS[label]
    pdfs = sorted(folder.glob("*.pdf"))
    if not pdfs:
        logger.warning("No PDFs found in %s", folder)
    return pdfs


def _get_interim_dir(label: str, pdf_stem: str | None = None) -> Path:
    """Return and ensure the interim directory for a company (and optionally a PDF)."""
    path = INTERIM_DIR / label
    if pdf_stem is not None:
        path = path / pdf_stem
    path.mkdir(parents=True, exist_ok=True)
    return path


def _table_pages_path(label: str, pdf_stem: str) -> Path:
    """Path to the table page indices JSON for a specific PDF."""
    return _get_interim_dir(label, pdf_stem) / "table_pages.json"


def _docling_json_path(label: str, pdf_stem: str) -> Path:
    """Path to the Docling extraction JSON for a specific PDF."""
    return _get_interim_dir(label, pdf_stem) / "docling.json"


def _chunks_parquet_path(label: str) -> Path:
    """Path to the combined chunks parquet for a company (all PDFs)."""
    return _get_interim_dir(label) / "chunks.parquet"


def _pdf_chunks_path(label: str, pdf_stem: str) -> Path:
    """Path to the per-PDF chunks parquet (crash-resilient intermediate)."""
    return _get_interim_dir(label, pdf_stem) / "chunks.parquet"


def _extract_table_context(text: str) -> str:
    """Extract ## headers from the top of Docling markdown to build a context prefix.

    Args:
        text: Docling markdown for a single page.

    Returns:
        A string like 'TABLE CONTEXT: FINANCIAL PERFORMANCE > SEGMENT OVERVIEW'
        or empty string if no headers found.
    """
    headers: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("## "):
            headers.append(stripped.lstrip("# ").strip())
        elif stripped.startswith("| ") or (stripped and not stripped.startswith("#")):
            # Stop once we hit table content or non-header prose
            break
    if not headers:
        return ""
    return "TABLE CONTEXT: " + " > ".join(headers)


def _resolve_labels(company: str | None) -> list[str]:
    """Resolve the --company option to a list of company labels."""
    if company is None:
        return list(COMPANY_FOLDERS.keys())
    if company not in COMPANY_FOLDERS:
        raise click.ClickException(
            f"Unknown company '{company}'. Choose from: {', '.join(COMPANY_FOLDERS)}"
        )
    return [company]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.group()
def cli() -> None:
    """Three-stage chunking pipeline for TPI Carbon Performance PDFs."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )


@cli.command()
@click.option(
    "--company",
    type=click.Choice(list(COMPANY_FOLDERS.keys())),
    default=None,
    help="Process a single company. Omit to process all.",
)
def classify(company: str | None) -> None:
    """Stage 1: Classify pages as table/non-table using numeric density.

    Reads all PDFs in data/raw/{company}/ with pdfplumber, computes numeric
    density per page, and writes table_pages.json per PDF to
    data/interim/{label}/{pdf_stem}/.
    """
    import pdfplumber
    from utils import NUMERIC_DENSITY_THRESHOLD, numeric_density

    labels = _resolve_labels(company)

    for label in labels:
        pdf_files = _get_pdf_files(label)
        click.echo(f"\n{'='*60}")
        click.echo(f"Classifying {label} — {len(pdf_files)} PDFs")
        click.echo(f"{'='*60}")

        for pdf_path in pdf_files:
            stem = pdf_path.stem
            out_path = _table_pages_path(label, stem)
            if out_path.exists():
                click.echo(f"\n  {stem} ... skipped (already classified)")
                continue
            click.echo(f"\n  {stem} ...")
            indices: list[int] = []

            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    words = page.extract_words()
                    density = numeric_density(words)
                    if density >= NUMERIC_DENSITY_THRESHOLD:
                        indices.append(i)

                total_pages = len(pdf.pages)

            out_path = _table_pages_path(label, stem)
            out_path.write_text(json.dumps(indices), encoding="utf-8")

            click.echo(
                f"    {len(indices)} table pages / {total_pages} total → {out_path.name}"
            )

    click.echo("\nClassification complete.")


@cli.command("extract-tables")
@click.option(
    "--company",
    type=click.Choice(list(COMPANY_FOLDERS.keys())),
    default=None,
    help="Process a single company. Omit to process all.",
)
def extract_tables(company: str | None) -> None:
    """Stage 2: Extract table pages with Docling. Run in docling_test env.

    Reads table_pages.json from Stage 1 for each PDF, converts only those
    pages with Docling, and writes docling.json per PDF to
    data/interim/{label}/{pdf_stem}/.
    """
    # Suppress HF symlink warning on Windows
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
    except ImportError:
        raise click.ClickException(
            "Docling is not installed. Run this command in the docling_test env:\n"
            "  conda activate docling_test && python chunking.py extract-tables"
        )

    # Disable OCR — these are text-layer PDFs
    pipeline_options = PdfPipelineOptions(do_ocr=False)
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )

    labels = _resolve_labels(company)

    for label in labels:
        pdf_files = _get_pdf_files(label)
        click.echo(f"\n{'='*60}")
        click.echo(f"Extracting tables for {label} — {len(pdf_files)} PDFs")
        click.echo(f"{'='*60}")

        for pdf_path in pdf_files:
            stem = pdf_path.stem

            out_path = _docling_json_path(label, stem)
            if out_path.exists():
                click.echo(f"\n  {stem} ... skipped (already extracted)")
                continue

            tp_path = _table_pages_path(label, stem)
            if not tp_path.exists():
                click.echo(
                    f"\n  SKIP {stem}: {tp_path} not found. Run 'classify' first.",
                    err=True,
                )
                continue

            table_page_indices: list[int] = json.loads(
                tp_path.read_text(encoding="utf-8")
            )

            if not table_page_indices:
                click.echo(f"\n  {stem}: 0 table pages — skipping Docling")
                # Write empty JSON so Stage 3 doesn't skip this PDF
                out_path = _docling_json_path(label, stem)
                out_path.write_text("[]", encoding="utf-8")
                continue

            click.echo(
                f"\n  {stem} — {len(table_page_indices)} table pages ..."
            )

            sorted_pages = sorted(table_page_indices)
            table_page_set = set(table_page_indices)
            page_dicts: list[dict] = []

            for i in range(0, len(sorted_pages), DOCLING_BATCH_SIZE):
                batch = sorted_pages[i : i + DOCLING_BATCH_SIZE]
                start_1idx = batch[0] + 1  # Docling pages are 1-indexed
                end_1idx = batch[-1] + 1

                click.echo(
                    f"    Batch {i // DOCLING_BATCH_SIZE + 1}: "
                    f"pages {start_1idx}–{end_1idx} "
                    f"({len(batch)} table pages) ..."
                )

                result = converter.convert(
                    str(pdf_path), page_range=(start_1idx, end_1idx)
                )
                doc = result.document

                for page_num in doc.pages:
                    # Docling page_num is 1-indexed; our indices are 0-indexed
                    if (page_num - 1) not in table_page_set:
                        continue
                    page_md = doc.export_to_markdown(page_no=page_num)
                    page_dicts.append({
                        "page_num": page_num,
                        "text": page_md,
                        "source": f"{label}/{stem}",
                        "is_table": True,
                    })

            out_path = _docling_json_path(label, stem)
            out_path.write_text(
                json.dumps(page_dicts, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            click.echo(f"    → {len(page_dicts)} table pages written")

    click.echo("\nDocling extraction complete.")


@cli.command()
@click.option(
    "--company",
    type=click.Choice(list(COMPANY_FOLDERS.keys())),
    default=None,
    help="Process a single company. Omit to process all.",
)
def chunk(company: str | None) -> None:
    """Stage 3: Chunk all pages and write combined chunks.parquet per company.

    Prose pages: Unstructured fast strategy → element_type chunking (header-prefixed).
    Table pages: Docling JSON with context prefix (whole-page chunks).
    Iterates all PDFs in data/raw/{company}/ and combines into one parquet.
    """
    import torch  # noqa: F401 — must load before numpy (Windows DLL order)
    import pandas as pd  # noqa: E402 — after torch to avoid fbgemm.dll conflict
    from unstructured.partition.pdf import partition_pdf

    labels = _resolve_labels(company)

    for label in labels:
        pdf_files = _get_pdf_files(label)
        click.echo(f"\n{'='*60}")
        click.echo(f"Chunking {label} — {len(pdf_files)} PDFs")
        click.echo(f"{'='*60}")

        combined_path = _chunks_parquet_path(label)
        if combined_path.exists():
            click.echo(f"  Skipped — {combined_path.name} already exists")
            continue

        for pdf_path in pdf_files:
            stem = pdf_path.stem

            # Skip PDFs already chunked (crash-resilient restart)
            pdf_out = _pdf_chunks_path(label, stem)
            if pdf_out.exists():
                click.echo(f"\n  {stem} ... skipped (already chunked)")
                continue

            docling_path = _docling_json_path(label, stem)
            if not docling_path.exists():
                click.echo(
                    f"\n  SKIP {stem}: {docling_path} not found. "
                    "Run 'extract-tables' first.",
                    err=True,
                )
                continue

            click.echo(f"\n  {stem} ...")

            # -- Load Docling table pages for this PDF --
            docling_pages: list[dict] = json.loads(
                docling_path.read_text(encoding="utf-8")
            )
            table_page_nums: set[int] = {
                p["page_num"] for p in docling_pages if p.get("is_table")
            }
            click.echo(f"    {len(docling_pages)} Docling table pages")

            # -- Partition with Unstructured (fast) --
            click.echo(f"    Partitioning with Unstructured (fast) ...")
            elements = partition_pdf(filename=str(pdf_path), strategy="fast")
            click.echo(f"    {len(elements)} elements returned")

            # -- Prose chunks: element_type (header-prefixed) --
            prose_elements = [
                el
                for el in elements
                if getattr(el.metadata, "page_number", None) not in table_page_nums
            ]
            click.echo(
                f"    {len(prose_elements)} prose elements "
                f"(excluded {len(elements) - len(prose_elements)} on table pages)"
            )

            source_label = f"{label}/{stem}"
            from utils import chunk_by_element_type

            prose_chunks = chunk_by_element_type(
                prose_elements,
                char_limit=CHAR_LIMIT,
                source_label=source_label,
            )

            # -- Table chunks: whole-page Docling + context prefix --
            table_chunks: list[dict] = []
            for idx, page in enumerate(docling_pages):
                if not page.get("is_table"):
                    continue
                text = page["text"].strip()
                if not text:
                    continue

                context_prefix = _extract_table_context(text)
                chunk_text = (
                    f"{context_prefix}\n\n{text}" if context_prefix else text
                )

                table_chunks.append({
                    "id": f"tbl_{stem}_{idx:04d}",
                    "text": chunk_text,
                    "strategy": "docling_table",
                    "source": source_label,
                    "page_num": page["page_num"],
                    "is_table": True,
                })

            pdf_chunks = prose_chunks + table_chunks
            click.echo(
                f"    {len(prose_chunks)} prose + {len(table_chunks)} table = "
                f"{len(pdf_chunks)} chunks"
            )

            # Save per-PDF parquet immediately
            if pdf_chunks:
                pd.DataFrame(pdf_chunks).to_parquet(pdf_out, index=False)
                click.echo(f"    → saved {pdf_out}")

        # -- Combine all per-PDF parquets into one company parquet --
        per_pdf_files = [
            _pdf_chunks_path(label, p.stem)
            for p in pdf_files
            if _pdf_chunks_path(label, p.stem).exists()
        ]
        if per_pdf_files:
            combined = pd.concat(
                [pd.read_parquet(f) for f in per_pdf_files],
                ignore_index=True,
            )
            combined.to_parquet(combined_path, index=False)
            click.echo(
                f"\n  Total: {len(combined)} chunks → {combined_path}"
            )
        else:
            click.echo(f"\n  WARNING: No chunks produced for {label}")

    click.echo("\nChunking complete.")


if __name__ == "__main__":
    cli()