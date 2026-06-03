# Diversified-Mining-RAG
A repository containing a Retrieval-Augmented-Generation (RAG) pipeline from corporate disclosure PDFs in the diversified mining sector. This project primarily aims to develop a tested best ingestion strategy.

## Overview

This pipeline ingests raw PDF disclosures (annual reports, sustainability reports, climate change reports) from two companies with contrasting approaches to GHG reporting:

- **Anglo American** provides ESG Factbooks with granular emissions breakdowns. The RAG pipeline focuses on qualitative methodological justifications and emissions targets expressed in associated reports.
- **ANTAM** discloses few (if any) emissions data. Queries should target disclosure _quality_ rather than precise emissions values.

The pipeline classifies PDF pages as table or non-table, extracts text using strategy-specific parsers (`unstructured` for prose, Docling for tables), chunks and embeds the content into ChromaDB, then retrieves and generates answers using an open-source LLM.

### Architecture

```
Raw PDFs → Classification → Extraction → Chunking → Embedding → Retrieval → Generation
             (pdfplumber)   (unstructured   (element_type   (MiniLM-L6-v2   (BM25+RRF    (Qwen2.5
                             + Docling)       + table ctx)    → ChromaDB)     +rerank)     0.5B-Instruct)
```

| Stage | Module | Output |
|-------|--------|--------|
| 1. Classify | `chunking.py classify` | `table_pages.json` per PDF |
| 2. Extract tables | `chunking.py extract-tables` | `docling.json` per PDF |
| 3. Chunk | `chunking.py chunk` | `chunks.parquet` per company |
| 4. Embed | `vector_store.py` | ChromaDB collection (`tpi_chunks`) |
| 5. Retrieve | `pipeline.py retrieve` | Top-k ranked chunks |
| 6. Generate | `pipeline.py generate` | Natural language answer |

## How to Run

### Prerequisites

- Python 3.11 (tested with 3.11.15)
- Conda (Miniconda or Anaconda)
- Windows (the `run_chunking.bat` script is Windows-specific)

### Setup

```bash
git clone git@github.com:elimossmarks11/Diversified-Mining-RAG.git
cd Diversified-Mining-RAG

# Create both conda environments
conda env create -f environment.yml          # creates 'rag'
conda env create -f docling_environment.yml  # creates 'docling_test'

conda activate rag
```

### Running the pipeline

1. Place raw PDFs in `data/raw/anglo/` and `data/raw/antam/`.

2. Run the full chunking pipeline (stages 1–3, handles environment switching automatically):
   ```bash
   run_chunking.bat
   ```

3. Embed chunks into ChromaDB (stage 4):
   ```bash
   python vector_store.py
   ```

4. Query via CLI (stages 5–6):
   ```bash
   # Retrieve only
   python pipeline.py retrieve "What are Anglo American's GHG emissions targets?"

   # Retrieve + generate
   python pipeline.py generate "What are Anglo American's GHG emissions targets?"
   ```

   Or run all stages end-to-end in a single command:
   ```bash
   python pipeline.py run "What are Anglo American's GHG emissions targets?"
   ```

### API server

```bash
python pipeline.py serve
```

Then query `POST http://127.0.0.1:8000/query` with `{"query": "...", "generate": true}`.
Swagger docs available at `http://127.0.0.1:8000/docs`.

## Configuration

Key parameters are defined as module-level constants in `utils.py`:

| Constant | Default | Purpose |
|----------|---------|---------|
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformer for chunk embeddings |
| `LLM_MODEL` | `Qwen/Qwen2.5-0.5B-Instruct` | Generation model |
| `TOP_K` | `5` | Number of chunks returned after reranking |
| `RRF_K` | `60` | Reciprocal Rank Fusion constant |
| `NUMERIC_DENSITY_THRESHOLD` | `0.06` | Page-level table classification threshold |

## Output

- **Intermediate outputs**: `data/interim/{company}/{report_name}/` — `table_pages.json`, `docling.json`, `chunks.parquet`
- **Vector store**: `data/interim/chroma/` — persisted ChromaDB collection
- **API responses**: JSON with retrieved chunks and optionally a generated answer

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for architecture details, chunking strategy evaluation, design decisions, and development setup.