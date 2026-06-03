"""Extract table pages with Docling. Run in docling_test conda env.

Reads {label}_table_pages.json (produced by chunking_evaluation.ipynb step 1)
to determine which pages to convert. Only converts identified table pages,
avoiding the cost of running Docling on the full document.
"""
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import json
import logging
from pathlib import Path

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Path(__file__) resolves relative to the script's own location,
# not the working directory — robust regardless of where you invoke Python from.
ROOT       = Path(__file__).resolve().parent.parent.parent
PDF_DIR    = ROOT / "data" / "chunking_test_pdfs"
OUTPUT_DIR = ROOT / "data" / "interim" / "docling_tests"

logger.info("ROOT:       %s", ROOT)
logger.info("PDF_DIR:    %s (exists=%s)", PDF_DIR, PDF_DIR.exists())
logger.info("OUTPUT_DIR: %s", OUTPUT_DIR)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DOCUMENTS = {
    "antam":         "ANTAM AR2016.pdf",
    "angloamerican": "AR2019.pdf",
}

# Max table pages per Docling call — limits peak RAM to avoid std::bad_alloc
# on large PDFs. Each batch converts at most this many pages at once.
DOCLING_BATCH_SIZE = 30


# Disable OCR — these are text-layer PDFs, OCR adds cost with no benefit
pipeline_options = PdfPipelineOptions(do_ocr=False)
converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
    }
)

for label, filename in DOCUMENTS.items():
    pdf_path = PDF_DIR / filename
    if not pdf_path.exists():
        logger.warning("SKIP %s — not found", pdf_path)
        continue

    # Read table page indices from classification step
    table_pages_path = OUTPUT_DIR / f"{label}_table_pages.json"
    if not table_pages_path.exists():
        logger.error(
            "%s not found. Run step 1 in chunking_evaluation.ipynb first.",
            table_pages_path,
        )
        continue

    table_page_indices = json.loads(table_pages_path.read_text(encoding="utf-8"))
    logger.info("Converting %s — %d table pages ...", label, len(table_page_indices))

    # Sort table pages and process in batches to avoid std::bad_alloc OOM.
    # Each batch uses page_range so Docling only preprocesses ~DOCLING_BATCH_SIZE
    # pages at a time instead of the full document.
    sorted_table_pages = sorted(table_page_indices)  # 0-indexed
    table_page_set = set(table_page_indices)
    page_dicts = []

    for i in range(0, len(sorted_table_pages), DOCLING_BATCH_SIZE):
        batch = sorted_table_pages[i : i + DOCLING_BATCH_SIZE]
        start_1idx = batch[0] + 1   # Docling pages are 1-indexed
        end_1idx   = batch[-1] + 1
        logger.info(
            "  Batch %d/%d: pages %d–%d (%d table pages in range)...",
            i // DOCLING_BATCH_SIZE + 1,
            -(-len(sorted_table_pages) // DOCLING_BATCH_SIZE),
            start_1idx,
            end_1idx,
            len(batch),
        )
        result = converter.convert(str(pdf_path), page_range=(start_1idx, end_1idx))
        doc = result.document

        for page_num, _page in doc.pages.items():
            # Docling page_num is 1-indexed; classification indices are 0-indexed
            if (page_num - 1) not in table_page_set:
                continue
            page_md = doc.export_to_markdown(page_no=page_num)
            page_dicts.append({
                "page_num": page_num,
                "text":     page_md,
                "source":   label,
                "is_table": True,
            })

    out_path = OUTPUT_DIR / f"{label}_docling.json"
    out_path.write_text(
        json.dumps(page_dicts, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("  Wrote %d table pages → %s", len(page_dicts), out_path)

logger.info("Done.")