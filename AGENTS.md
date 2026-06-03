# Agent Instructions - DS205 PS2

This file provides instructions for Github Copilot. It is largely inspired by [this github repo](https://github.com/meleantonio/ChernyCode/blob/main/AGENTS.md). 

## Project Context

This repository contains a RAG pipeline for extracting Carbon Performance information from corporate disclosure PDFs for the Diversified Mining sector (ANTAM and Anglo American), built for DS205 at LSE.

### Pipeline Architecture

| Stage | Description | Output |
|-------|-------------|--------|
| **Classification** | Extract words with pdfplumber, compute numeric density per page. Classify pages as table or non-table. | Page-level table/non-table labels |
| **Extraction** | Non-table pages: `unstructured` fast strategy. Table pages: Docling (OCR off, `do_table_structure=False`). | Per-page text/markdown |
| **Chunking** | Segment extracted text into semantically coherent chunks with metadata (company, year, page, source strategy). | `chunks.parquet` |
| **Embedding** | Encode chunks using `sentence-transformers/all-MiniLM-L6-v2`. | Vector embeddings in ChromaDB |
| **Retrieval** | Query ChromaDB for top-k relevant chunks given a natural language question. | Ranked chunk list with scores |
| **Generation** | Feed retrieved chunks to an open-source HuggingFace LLM to produce answers. | Natural language answer |

## Required Libraries

The following libraries are approved for this project. Course-required libraries are marked; others are justified in `CONTRIBUTING.md`.

| Library | Purpose | Status |
|---------|---------|--------|
| `unstructured` | PDF parsing, element-type chunking | Course-required |
| `sentence-transformers` | Embedding model inference | Course-required |
| `torch` (CPU-only) | ML backend | Course-required |
| `chromadb` | Vector store for retrieval | Course-required |
| `gensim` | Word2Vec baseline embeddings | Course-required |
| `click` | CLI interface | Course-required |
| `pdfplumber` | Spatial PDF extraction | Justified in CONTRIBUTING.md |
| `docling` | Vision-based PDF extraction | Justified in CONTRIBUTING.md |

**Adding new libraries:** Do not introduce a new library without empirical evidence (benchmark, comparison, or evaluation metric) demonstrating it outperforms the current approach. Document the justification in `CONTRIBUTING.md`.

## Coding Standards

### Python Development
- Use Python 3.10+
- Format with Ruff (`ruff format .`)
- Lint with Ruff (`ruff check .`)
- Always include type hints
- Use Google-style docstrings
- Follow PEP 8 with max line length 120
- Each function has to do one thing only
- When several functions have the same arguments, create a dataclass called `RetrievalConfig`
- Don't duplicate code. Ensure only one instance of an object is created and shared across the program
- Unless lazy imports are required, all libraries should only be imported once, at the top of the .ipynb or .py file.

### No Hardcoded Values
- Never hardcode thresholds, paths, or magic numbers inline
- Extract constants to module-level `UPPER_SNAKE_CASE` variables with a documenting comment
- If a value truly cannot be parameterised, it must have a comment explaining **why** it is hardcoded

### Logging
- Use the `logging` module for all library/module code (`pipeline.py`, `chunking.py`, `utils.py`)
- Use `click.echo()` **only** inside CLI entry points (Click command functions)
- Never use bare `print()` for operational output

### Naming Conventions
- Functions/variables: snake_case
- Classes: PascalCase
- Constants: UPPER_SNAKE_CASE
- Files: snake_case.py

## Module Boundaries

Each module has a single responsibility. Do not leak logic across boundaries.

| Module | Responsibility | Owns |
|--------|---------------|------|
| `pipeline.py` | CLI entry points and orchestration. Calls helpers in sequence. | Click commands, file I/O orchestration, progress reporting |
| `utils.py` | All stateless helpers: extraction, chunking, embedding, retrieval, evaluation. One function per pipeline stage. | PDF parsing, chunking, embedding wrappers, ChromaDB queries, evaluation metrics |

**Rules:**
- Functions must be idempotent — calling them twice with the same input produces the same output
- `utils.py` must not import from `pipeline.py`; shared configuration flows downward from `pipeline.py` via arguments or a `RetrievalConfig` dataclass
- If `utils.py` grows unwieldy, split by concern (e.g., `extraction.py` for PDF parsing/chunking). Verify before doing this.

## Data Directory Conventions

```
data/
├── raw/                    # Original unmodified PDFs, organised by company
│   ├── antam/
│   └── anglo_american/
├── interim/                # Intermediate processing outputs (extractable, reproducible)
│   ├── {company}/
│   │   ├── chunks.parquet
│   │   └── processed_files.json
│   └── docling_tests/      # Extraction method comparisons
└── processed/              # Final evaluation-ready outputs
```

- Never modify files in `data/raw/`
- Intermediate outputs go in `data/interim/`
- Final outputs go in `data/processed/`

## Notebooks vs Scripts

- **Notebooks** (`.ipynb`) are sandboxes for experimentation and exploration only
- **Production code** lives exclusively in `.py` modules
- Never put pipeline logic in a notebook; extract proven approaches into the appropriate module

## Testing Strategy

- Testing is **evaluation-based**: measure retrieval quality (recall, precision, MRR) against `ground_truth.json`
- `ground_truth.json` contains curated queries with expected text snippets
- Validate changes by running the evaluation and comparing metrics before and after

## Workflow Guidelines

### Planning
- Never commit or edit files without securing confirmation first

### Error Handling
- Use try-except with proper logging (`logging` module)
- Provide clear error messages
- Don't silently ignore exceptions

## Key Files

| File | Purpose |
|------|---------|
| `AGENTS.md` | This file — Github Copilot instructions |
| `pipeline.py` | CLI entry point and pipeline orchestration |
| `chunking.py` | Chunking strategies and table detection |
| `utils.py` | Embedding, retrieval, and evaluation helpers |
| `ground_truth.json` | Evaluation queries with expected snippets |
| `CONTRIBUTING.md` | Library justifications and architecture docs |

## Tools and Commands

| Task | Command |
|------|---------|
| Format code | `ruff format .` |
| Lint code | `ruff check .` |
| Fix lint issues | `ruff check --fix .` |
| Activate environment | `conda activate rag` |
| Update environment | `conda env update -n rag -f environment.yml --prune` |

## Preferences

- Provide concise, focused responses
- Show code examples when helpful
- Explain the "why" behind changes
- Prefer editing existing files over creating new ones
- Only create documentation when explicitly requested