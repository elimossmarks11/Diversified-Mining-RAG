# Contributing

Thank you for your interest in contributing to this project! This document is aimed at developers who want to understand the pipeline internals, fix bugs, or extend the codebase.

## Most valuable contributions

This pipeline has focussed most closely on developing an empirically-tested chunking strategy, with a particular focus on table extraction. The `testing` folder contains several `.ipynb` files that list different approaches to the unsolved problem of pdf extraction. When testing the generation stage, this chunking strategy proved successful but was let down by the model - better model testing would be a key contribution to this project.

## How the Pipeline Works

The pipeline has six stages, split across two conda environments due to a dependency conflict between `unstructured` and `docling` (incompatible `transformers` versions).

### Architecture overview

```
┌─────────────────────────────────────────────────────────────────┐
│  run_chunking.bat  (automated env switching)                    │
│                                                                 │
│  Stage 1: classify    ──▶  table_pages.json per PDF   [rag]    │
│  Stage 2: extract-tables ▶ docling.json per PDF  [docling_test]│
│  Stage 3: chunk       ──▶  chunks.parquet per company [rag]    │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  python vector_store.py   [rag]                                 │
│  Stage 4: embed       ──▶  ChromaDB at data/interim/chromadb/  │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  python pipeline.py retrieve / generate / serve   [rag]         │
│  Stage 5: retrieve    ──▶  top-k chunks (dense+BM25+RRF+CE)   │
│  Stage 6: generate    ──▶  Qwen2.5-0.5B-Instruct answer       │
└─────────────────────────────────────────────────────────────────┘
```

### Module responsibilities

| Module | Stage | Responsibility |
|--------|-------|----------------|
| `chunking.py` | 1–3 | Click CLI group with `classify`, `extract-tables`, and `chunk` subcommands. Classification uses pdfplumber numeric density (threshold calculated on test data at 0.06). Table extraction uses Docling (OCR off). Prose chunking uses `unstructured` fast strategy with `element_type` header-prefixed chunks. |
| `vector_store.py` | 4 | Reads `chunks.parquet` per company, embeds with `all-MiniLM-L6-v2`, upserts into a single ChromaDB collection (`tpi_chunks`). |
| `utils.py` | — | All stateless helpers: chunking strategies, embedding wrappers, BM25/RRF/reranking, query expansion, prompt building, generation, CuEq calculation. No CLI code. Some helpers are not used in the pipeline but are important for the testing notebooks so remain in case of further testing. |
| `retrieval.py` | 5 | Standalone debug script for hybrid retrieval. |
| `generation.py` | 6 | Standalone debug script for retrieval + generation. |
| `pipeline.py` | 1–6 | Click CLI orchestrator. Delegates chunking/embedding to subprocess calls, retrieval/generation to `utils.py` functions. Also hosts `serve` command for the FastAPI server. |
| `app.py` | 5–6 | FastAPI application. `POST /query` accepts any natural-language question, returns retrieved chunks and optionally an LLM-generated answer. |

### Data flow

1. **Classification** (`chunking.py classify`): pdfplumber opens each PDF in `data/raw/{company}/`, computes numeric density per page, writes `table_pages.json` (list of 0-indexed page indices) to `data/interim/{company}/{pdf_stem}/`.

2. **Table extraction** (`chunking.py extract-tables`, docling_test env): Reads `table_pages.json`, runs Docling (OCR disabled) on those pages only, writes `docling.json` per PDF.

3. **Chunking** (`chunking.py chunk`): For each PDF, non-table pages are parsed by `unstructured` (fast strategy) and chunked with `element_type` header-prefixed logic. Table pages use Docling markdown as whole-page chunks with a `TABLE CONTEXT:` prefix. All chunks are saved as `chunks.parquet` per company.

4. **Embedding** (`vector_store.py`): Loads parquet files, embeds all chunks with `all-MiniLM-L6-v2` (384-dim, L2-normalised), upserts into ChromaDB with metadata (company, source, pages, strategy).

5. **Retrieval** (`utils.retrieve_chunks()`): Dense bi-encoder retrieval (top-50) → query expansion for domain terms ("activity" → production volume variants) → BM25 over combined candidate pool → Reciprocal Rank Fusion across all ranked lists → cross-encoder reranking (top-5).

6. **Generation** (`utils.generate_answer()`): Trims retrieved chunks to fit a 2048-token budget, builds a chat-template prompt (system message with domain context + sources + question), generates with `Qwen/Qwen2.5-0.5B-Instruct` (greedy decoding, repetition_penalty=1.2).

### Key design decisions

1. **Two conda environments**: `unstructured` and `docling` require incompatible `transformers` versions. The `run_chunking.bat` script automates the env switch between stages. This is the only Windows-specific requirement.

2. **Hybrid retrieval**: Pure dense retrieval with MiniLM struggles on table chunks (markdown tables have low semantic similarity to natural-language queries). BM25 + RRF fusion recovers table chunks that dense retrieval misses.

3. **Query expansion**: Domain-specific terms like "activity" (which means "production volume in copper equivalent tonnes" in the TPI framework) are expanded into multiple retrieval queries and fused via RRF, bridging the vocabulary gap between questions and corporate disclosure text.

4. **Qwen2.5-0.5B-Instruct over TinyLlama/distilgpt2**: distilgpt2 (82M) produced degenerate repetition loops. TinyLlama (1.1B) failed to load on the development machine. Qwen (0.5B) is instruction-tuned with chat template support and produces structured, cited answers with `repetition_penalty=1.2`. Heavier models that are too slow for the CPU environment this was tested on were abandoned.

### Chunking Strategy

Six chunking strategies were evaluated on one Anglo American disclosure PDF (AR 2019) against 6 ground-truth queries (5 prose, 1 table) using recall@5 and MRR with `sentence-transformers/all-MiniLM-L6-v2` embeddings.

**Winner: `element_type` chunking (1.0 recall@5, 0.36 MRR dense / 0.53 MRR hybrid)**

The `element_type` strategy segments text at document structure boundaries (headers → headers) and prepends each chunk with a context prefix derived from the running title and section header:

```
RUNNING TITLE: <latest document title>
HEADER (H2): <current section header>
<chunk body text>
```

This outperforms all alternatives because the context prefix gives MiniLM a semantic bridge between the query topic and the chunk content. Strategies without prefixes (`char_limit`, `token_aware`, `table_aware`) failed on queries targeting specific document sections (energy targets, GHG targets) where the answer text alone lacked sufficient topical signal.

| Strategy | Recall@5 | MRR | Why it underperforms |
|----------|----------|-----|----------------------|
| `element_type` | 1.0 | 0.36 | — |
| `element_type_hybrid` | 1.0 | 0.53 | — (best MRR with BM25 fusion) |
| `paragraph` | 0.8 | 0.42 | Mechanical `\n\n` splits break topic coherence |
| `token_aware` | 0.8 | 0.54 | No header prefix; misses section-specific queries |
| `char_limit` | 0.6 | 0.29 | Fixed-width windows ignore structure |
| `sentence_sliding` | 0.4 | 0.33 | Windows too small (~99 tokens) to capture full statements |
| `table_aware` | 0.0 | 0.0 | Row-level linearization fragmented prose chunk matches |

**Table retrieval limitation:** Dense retrieval with MiniLM cannot bridge the gap between natural-language queries and markdown table rows. A table chunk at BM25 rank 2 landed at dense rank 90 (sim=0.33). Hybrid retrieval (BM25 + MiniLM via Reciprocal Rank Fusion) improves MRR but cannot overcome this when ~5 prose chunks rank well on both signals. Linearizing table rows into natural language (e.g., `"De Beers: Underlying EBIT = 4,149"`) was tested but broke snippet matching for existing queries when applied naively via row-level splitting.

**Production chunking method for `chunking.py`:**
- Non-table pages: `element_type` chunking with header context prefixes (from `unstructured` extraction)
- Table pages: Docling extraction, kept as whole-page chunks with table context prefix (from `##` headers). A lightweight programmatic solution (reconstructing tables based on spatial clustering) was tested but results were far poorer than Docling extraction.
- Retrieval: hybrid (MiniLM dense + BM25, fused via RRF)

### Embedding Model

When tested on Anglo American's AR 2019, MiniLM-L6-v2 outperformed harrier-oss-v1-270m (a high-performing small model on MTEB leaderboard). 
harrier-oss-v1-270m was evaluated both with and without the query-side instruction prefix (`prompt_name="web_search_query"`) which the model expects. Prompted harrier did not consistently outperform unprompted harrier, and neither configuration matched MiniLM hybrid. MiniLM was selected on the basis of superior Recall@5 and simpler deployment.

| model | method | recall@5 | mrr |
|---|---|---|---|
| MiniLM-L6-v2 | hybrid | 0.549 | 0.506 |
| harrier-270m (no prompt) | hybrid | 0.417 | 0.333 |
| harrier-270m (prompted) | dense | 0.417 | 0.292 |
| harrier-270m (prompted) | hybrid | 0.389 | 0.500 |
| MiniLM-L6-v2 | dense | 0.382 | 0.297 |
| harrier-270m (no prompt) | dense | 0.250 | 0.250 |

#### Evaluation

| Question |  Facts in chunks? | Facts in answer? | Problem type |
|----------|------------------|------------------|--------------|
| Emissions targets | Yes | Yes | None | 
| Activity of emissions | Yes | Partially - Qwen picked up the production output for one mine when asked about the total amount | Generation | 
| Changed over time? | Yes | No | Generation |

## Known Bugs / Areas for Improvement

1. The table detection heuristic (`is_table_page`) identifies tables according to numerical density. This accurately captures tables (two tests on ANTAM and Anglo American documents reported 97% accuracy) but can miss pages that are split between narrative text and tables. A region-level detection method was attempted, but it proved too sensitive to reliably use. This is not a bug because tables that are not detected are not abandoned, but `unstructured` fast strategy will not render them cleanly for LLM evaluation.

2. I set RRF_K = 60 in line with Cormack et al. (2009). My evaluation set of queries was too small (6) to tune this value to my corpora but a larger evaluation could result in a more reflective value. 

3. Even with query expansion, Qwen frequently picks up on the wrong production statistic when queried about emissions activity. 

4. The TPI's emissions intensity calculation is not provided by this pipeline. To do so, a separate `.py` script that augments raw production data with World Bank historical data would be necessary, which the current generation model is incapable of running independently. 

## Setting Up the Development Environment

- On Windows, `torch` occasionally fails to import with `OSError: [WinError 127]` 
  related to `fbgemm.dll`. A full machine restart resolves this. If it persists,
  reinstall the Visual C++ redistributable from https://aka.ms/vs/17/release/vc_redist.x64.exe
  and restart again.

### Prerequisites

- **Python 3.11** (tested with 3.11.15)
- **Conda** (Miniconda or Anaconda) for environment management
- **Windows** (the `run_chunking.bat` script and `KMP_DUPLICATE_LIB_OK` workaround are Windows-specific; macOS/Linux users would need to adapt the bat script to a shell script)

**`rag` environment** (main pipeline — chunking stages 1 & 3, embedding, retrieval, generation, API):

| Package | Version |
|---------|---------|
| torch | 2.5.1+cpu |
| transformers | 4.57.6 |
| sentence-transformers | 5.3.0 |
| chromadb | 1.5.5 |
| unstructured | 0.21.5 |
| pdfplumber | 0.11.9 |
| gensim | 4.4.0 |
| click | 8.3.1 |
| fastapi | 0.135.3 |
| uvicorn | 0.41.0 |
| rank-bm25 | 0.2.2 |
| numpy | 2.4.3 |
| pandas | 3.0.1 |

**`docling_test` environment** (chunking stage 2 only — table extraction):

| Package | Version |
|---------|---------|
| docling | 2.82.0 |
| torch | 2.11.0 |
| transformers | 4.57.6 |
| click | 8.3.1 |
| pandas | 3.0.1 |

### Installation

```bash
# 1. Clone the repository
git clone git@github.com:lse-ds205/problem-set-2-elimossmarks11.git
cd problem-set-2-elimossmarks11

# 2. Create both conda environments
conda env create -f environment.yml          # creates 'rag'
conda env create -f docling_environment.yml  # creates 'docling_test'

# 3. Activate the main environment
conda activate rag
```

### Running the pipeline

#### Prerequisites

1. Place raw PDFs in `data/raw/`, organised by company:
   ```
   data/raw/
   ├── anglo/          # Anglo American PDFs
   └── antam/          # ANTAM PDFs
   ```
2. Activate the main environment: `conda activate rag`

#### Stage 1–3: Chunking (classify → extract → chunk)

The batch script handles all three stages and switches between conda environments
automatically (the `docling_test` env is used for table extraction only):

```bash
run_chunking.bat
```

This produces `chunks.parquet` files under `data/interim/{company}/{report_name}/`.

To run chunking stages individually via the CLI orchestrator:

```bash
python pipeline.py chunk
```

#### Stage 4: Embedding

Embed all chunks into ChromaDB:

```bash
python vector_store.py
```

This reads every `chunks.parquet` in `data/interim/` and upserts embeddings into the
`tpi_chunks` collection (persisted in `data/interim/chroma/`).

#### Stage 5–6: Retrieval and generation

**Retrieve only** (returns top-k chunks without generation):

```bash
python pipeline.py retrieve "What are Anglo American's GHG emissions targets?"
```

**Retrieve + generate** (feeds retrieved chunks to Qwen 2.5-0.5B-Instruct):

```bash
python pipeline.py generate "What are Anglo American's GHG emissions targets?"
```

**All-in-one** (chunk → embed → retrieve → generate in a single command):

```bash
python pipeline.py run "What are Anglo American's GHG emissions targets?"
```

#### API server

Launch the FastAPI server:

```bash
python pipeline.py serve
```

Query the API (PowerShell):

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/query `
    -Method POST `
    -ContentType "application/json" `
    -Body '{"query": "What are Anglo American''s GHG emissions targets?", "generate": true}'
```

Or with curl:

```bash
curl -X POST http://127.0.0.1:8000/query \
    -H "Content-Type: application/json" \
    -d '{"query": "What are Anglo American'\''s GHG emissions targets?", "generate": true}'
```

Health check: `GET http://127.0.0.1:8000/health`
Swagger docs: `http://127.0.0.1:8000/docs`

## Code Style and Guidelines

Follow all coding conventions outlined in the repository's `AGENTS.md`.