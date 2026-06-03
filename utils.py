"""
W09 shared utilities — chunking, embedding, evaluation helpers.

Used by the W09 lecture notebook and available for PS2 scripts.
"""

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

# Workaround: conda numpy (MKL) and pip torch ship conflicting OpenMP
# runtimes on Windows. torch must load before numpy/pandas to claim DLLs first.
if os.name == "nt":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch  # noqa: E402 — must precede numpy/pandas on Windows (DLL conflict)

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from gensim.utils import simple_preprocess
from nltk.tokenize import sent_tokenize

if TYPE_CHECKING:
    import chromadb
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Numeric density above which a page is classified as a table page.
# Calibrated on ANTAM AR2016 + Anglo American AR2019 (F1=0.94, recall=1.00).
# See testing/table_extraction_tests/threshold_calibration.ipynb
NUMERIC_DENSITY_THRESHOLD = 0.06

# Pattern matching purely numeric tokens (e.g. "1,234", "56.78")
NUMERIC_RE = re.compile(r"\b\d+([.,]\d+)*\b")

# ---------------------------------------------------------------------------
# Table detection
# ---------------------------------------------------------------------------


def numeric_density(words: list[dict]) -> float:
    """Fraction of pdfplumber words that are purely numeric.

    Args:
        words: List of word dicts from pdfplumber's extract_words().

    Returns:
        Proportion of words matching a numeric pattern, between 0.0 and 1.0.
    """
    if not words:
        return 0.0
    texts = [w["text"] for w in words]
    hits = sum(1 for t in texts if NUMERIC_RE.fullmatch(t.strip(",.")))
    return hits / len(texts)


# ---------------------------------------------------------------------------
# Environment and config
# ---------------------------------------------------------------------------


def require_env(name: str) -> str:
    """Return a required environment variable value or raise a clear setup error."""
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(
            f"{name} is not set. Copy .env.example to .env and set {name}."
        )
    return value


def bootstrap_runtime_env(dotenv_path: str = ".env") -> None:
    """Load .env and apply runtime environment defaults safely."""
    load_dotenv(dotenv_path=dotenv_path, encoding="utf-8-sig")

    if os.getenv("HF_HOME", "").strip():
        os.environ["HF_HOME"] = os.getenv("HF_HOME", "").strip()

    if os.name == "nt":
        os.environ["KMP_DUPLICATE_LIB_OK"] = os.getenv("KMP_DUPLICATE_LIB_OK", "TRUE")


def resolve_pdf_workflow_config() -> dict:
    """Resolve runtime configuration for PDF parsing and retrieval steps."""
    bootstrap_runtime_env()

    pdf_dir = Path(require_env("PDF_SOURCE_DIR"))
    if not pdf_dir.is_dir():
        raise ValueError(f"PDF_SOURCE_DIR is not a valid folder: {pdf_dir}")

    pdf_glob = os.getenv("PDF_GLOB", "*.pdf").strip() or "*.pdf"
    pdf_candidates = sorted(pdf_dir.glob(pdf_glob))
    if not pdf_candidates:
        raise ValueError(f"No PDFs found in {pdf_dir} with pattern: {pdf_glob}")

    chroma_dir = Path(os.getenv("CHROMA_DIR", "data/chromadb").strip())

    return {
        "pdf_dir": pdf_dir,
        "pdf_glob": pdf_glob,
        "pdf_candidates": pdf_candidates,
        "pdf_path": pdf_candidates[0],
        "hf_home": os.environ.get("HF_HOME", "(default HuggingFace cache)"),
        "kmp": os.environ.get("KMP_DUPLICATE_LIB_OK"),
        "query_text": os.getenv(
            "QUERY_TEXT",
            "What are Ajinomoto's Scope 1 and 2 emissions targets?",
        ),
        "output_dir": Path(os.getenv("OUTPUT_DIR", "data/interim")),
        "chroma_dir": chroma_dir,
    }


# ---------------------------------------------------------------------------
# Chunking strategies
# ---------------------------------------------------------------------------


def chunk_by_char_limit(
    raw_texts: list[str],
    char_limit: int = 1000,
    source_label: str = "",
) -> list[dict]:
    """Chunk consecutive text elements by cumulative character limit.

    This is a sequential baseline: it does not detect true paragraph
    boundaries; it just accumulates adjacent text until *char_limit*.
    """
    chunks: list[dict] = []
    buffer: list[str] = []
    buffer_len = 0

    for text in raw_texts:
        if buffer_len + len(text) > char_limit and buffer:
            chunks.append(
                {
                    "id": f"char_{len(chunks):04d}",
                    "text": " ".join(buffer),
                    "strategy": "char_limit",
                    "source": source_label,
                }
            )
            buffer = []
            buffer_len = 0

        buffer.append(text)
        buffer_len += len(text)

    if buffer:
        chunks.append(
            {
                "id": f"char_{len(chunks):04d}",
                "text": " ".join(buffer),
                "strategy": "char_limit",
                "source": source_label,
            }
        )

    return chunks


def chunk_by_element_type(
    elements: list,
    char_limit: int = 1000,
    source_label: str = "",
) -> list[dict]:
    """Group elements into header-bounded chunks with explicit context prefixes.

    Each chunk follows this shape:
    RUNNING TITLE: <latest Title>

    HEADER (H2): <current Header>

    <body text between this header and the next one>
    """
    # -- pass 1: build header-delimited sections with running-title context --
    sections: list[dict] = []

    running_title = ""
    section_title = ""
    section_header = ""
    section_body: list[str] = []
    section_pages: set = set()
    section_types: set = set()

    for el in elements:
        text = (getattr(el, "text", None) or "").strip()
        if not text:
            continue

        el_type = type(el).__name__
        page = getattr(el.metadata, "page_number", None)

        if el_type == "Title":
            running_title = text
            continue

        if el_type == "Header":
            if section_header and section_body:
                sections.append(
                    {
                        "running_title": section_title,
                        "header": section_header,
                        "body": section_body,
                        "pages": section_pages,
                        "element_types": section_types,
                    }
                )

            section_title = running_title
            section_header = text
            section_body = []
            section_pages = {page} if page else set()
            section_types = {"Header"}
            if section_title:
                section_types.add("Title")
            continue

        if not section_header:
            section_title = running_title
            section_header = "(no header)"
            section_body = []
            section_pages = set()
            section_types = set()
            if section_title:
                section_types.add("Title")

        section_body.append(text)
        section_types.add(el_type)
        if page:
            section_pages.add(page)

    if section_header and section_body:
        sections.append(
            {
                "running_title": section_title,
                "header": section_header,
                "body": section_body,
                "pages": section_pages,
                "element_types": section_types,
            }
        )

    # -- pass 2: emit chunks; never emit heading-only chunks --
    chunks: list[dict] = []

    for section in sections:
        body_texts = section["body"]
        if not body_texts:
            continue

        running_title_text = section["running_title"]
        header_text = section["header"]

        prefix_parts = []
        if running_title_text:
            prefix_parts.append(f"RUNNING TITLE: {running_title_text}")
        prefix_parts.append(f"HEADER (H2): {header_text}")

        prefix = "\n\n".join(prefix_parts)
        available = char_limit - len(prefix) - 2
        if available < 100:
            available = max(char_limit // 2, 100)

        body_buffer: list[str] = []
        body_buffer_len = 0

        for body_piece in body_texts:
            if body_buffer_len + len(body_piece) > available and body_buffer:
                chunks.append(
                    {
                        "id": f"elem_{len(chunks):04d}",
                        "text": f"{prefix}\n\n" + "\n".join(body_buffer),
                        "strategy": "element_type",
                        "source": source_label,
                        "pages": sorted(section["pages"]),
                        "element_types": sorted(section["element_types"]),
                    }
                )
                body_buffer = []
                body_buffer_len = 0

            body_buffer.append(body_piece)
            body_buffer_len += len(body_piece)

        if body_buffer:
            chunks.append(
                {
                    "id": f"elem_{len(chunks):04d}",
                    "text": f"{prefix}\n\n" + "\n".join(body_buffer),
                    "strategy": "element_type",
                    "source": source_label,
                    "pages": sorted(section["pages"]),
                    "element_types": sorted(section["element_types"]),
                }
            )

    return chunks


def chunk_sentences_sliding(
    texts: list[str],
    window_size: int = 5,
    overlap: int = 2,
    source_label: str = "",
) -> list[dict]:
    """Split texts into sentence-based sliding window chunks.

    Tokenise all texts into sentences, then slide a window of *window_size*
    sentences with *overlap* shared sentences between consecutive chunks.
    """
    all_sentences: list[str] = []
    for text in texts:
        all_sentences.extend(sent_tokenize(text))

    chunks: list[dict] = []
    step = max(1, window_size - overlap)
    for i in range(0, len(all_sentences), step):
        window = all_sentences[i : i + window_size]
        chunk_text = " ".join(window)
        if len(chunk_text.strip()) < 20:
            continue
        chunks.append(
            {
                "id": f"sent_{i:04d}",
                "text": chunk_text,
                "strategy": "sentence_window",
                "source": source_label,
                "sentence_start": i,
                "sentence_end": i + len(window) - 1,
            }
        )
    return chunks


def save_chunks_as_markdown(
    chunks_df: pd.DataFrame, path: Path | str, strategy_name: str
) -> None:
    """Save chunks to a readable Markdown file for manual review."""
    path = Path(path)

    if "char_count" not in chunks_df.columns:
        chunks_df = chunks_df.copy()
        chunks_df["char_count"] = chunks_df["text"].fillna("").str.len()

    lines = [
        f"# {strategy_name}",
        "",
        f"Total chunks: {len(chunks_df)}  ",
        f"Mean chars: {chunks_df['char_count'].mean():.0f}  ",
        f"Median chars: {chunks_df['char_count'].median():.0f}",
        "",
        "---",
        "",
    ]

    for _, row in chunks_df.iterrows():
        lines.append(f"## {row['id']}")

        if "pages" in row.index and row["pages"]:
            lines.append(f"**Pages:** {row['pages']}  ")
        if "element_types" in row.index and row["element_types"]:
            lines.append(f"**Element types:** {row['element_types']}  ")

        lines.append(f"**Chars:** {row['char_count']}")
        lines.append("")
        lines.append("```")
        lines.append(str(row["text"]))
        lines.append("```")
        lines.append("")
        lines.append("---")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------


def embed_chunks_w2v(chunks: list[dict], model) -> np.ndarray:
    """Embed chunks by averaging Word2Vec word vectors."""
    vecs = []
    zero_count = 0
    for chunk in chunks:
        tokens = simple_preprocess(chunk["text"])
        word_vecs = [model.wv[t] for t in tokens if t in model.wv]
        if word_vecs:
            vecs.append(np.mean(word_vecs, axis=0))
        else:
            vecs.append(np.zeros(model.wv.vector_size))
            zero_count += 1
    if zero_count:
        logger.warning("Chunks with all-zero embeddings: %d/%d", zero_count, len(chunks))
    return np.array(vecs)


def count_tokens(texts: list[str], tokenizer) -> list[int]:
    """Count tokens for each text using the given HF tokenizer."""
    return [len(tokenizer.encode(t, add_special_tokens=False)) for t in texts]


def load_embedding_model(model_name: str) -> "SentenceTransformer":
    """Load a SentenceTransformer model on CPU.

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        Loaded SentenceTransformer instance.
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device="cpu")


def embed_texts(
    model: "SentenceTransformer",
    texts: list[str],
    batch_size: int = 256,
) -> list[list[float]]:
    """Encode texts in batches with L2 normalisation.

    Args:
        model: A loaded SentenceTransformer model.
        texts: List of text strings to embed.
        batch_size: Number of texts per encoding batch.

    Returns:
        Nested list of embedding vectors (one per text).
    """
    total_batches = (len(texts) + batch_size - 1) // batch_size
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        embs = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        all_embeddings.extend(embs.tolist())
        logger.info("Embedded batch %d/%d", i // batch_size + 1, total_batches)
    return all_embeddings


def get_chroma_client(chroma_dir: Path) -> "chromadb.PersistentClient":
    """Create or open a persistent ChromaDB client.

    Args:
        chroma_dir: Directory for ChromaDB storage files.

    Returns:
        A PersistentClient instance.
    """
    import chromadb

    chroma_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(chroma_dir))


def upsert_to_collection(
    collection: "chromadb.Collection",
    ids: list[str],
    embeddings: list[list[float]],
    documents: list[str],
    metadatas: list[dict],
    batch_size: int = 256,
) -> None:
    """Upsert embeddings into a ChromaDB collection in batches.

    Args:
        collection: Target ChromaDB collection.
        ids: Chunk identifiers.
        embeddings: Pre-computed embedding vectors.
        documents: Raw text documents.
        metadatas: Per-chunk metadata dicts.
        batch_size: Number of records per upsert call.
    """
    for i in range(0, len(ids), batch_size):
        end = i + batch_size
        collection.upsert(
            ids=ids[i:end],
            embeddings=embeddings[i:end],
            documents=documents[i:end],
            metadatas=metadatas[i:end],
        )


# ---------------------------------------------------------------------------
# Exploration and search helpers
# ---------------------------------------------------------------------------


def find_chunks_containing(
    keyword: str,
    chunks: list[dict],
    max_results: int = 5,
) -> list[str]:
    """Return IDs of chunks whose text contains *keyword* (case-insensitive)."""
    hits = [c for c in chunks if keyword.lower() in c["text"].lower()]
    for c in hits[:max_results]:
        logger.info("  %s: %s...", c["id"], c["text"][:150])
    if not hits:
        logger.info("  No chunks contain '%s'", keyword)
    return [c["id"] for c in hits[:max_results]]


def rank_by_model(
    chunks: list[dict], query: str, model, model_type: str
) -> pd.DataFrame:
    """Embed chunks and query with the given model, return ranked results DataFrame."""
    if model_type == "sentence_transformer":
        chunk_embeddings = model.encode(
            [c["text"] for c in chunks], normalize_embeddings=True
        )
        query_vec = model.encode([query], normalize_embeddings=True)[0]
        similarities = np.dot(chunk_embeddings, query_vec)
    elif model_type == "word2vec":
        chunk_embeddings = embed_chunks_w2v(chunks, model)
        query_vec = embed_chunks_w2v([{"text": query}], model)[0]
        norms_chunk = np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)
        norms_chunk[norms_chunk == 0] = 1
        chunk_normed = chunk_embeddings / norms_chunk
        q_norm = np.linalg.norm(query_vec)
        query_normed = query_vec / q_norm if q_norm > 0 else query_vec
        similarities = np.dot(chunk_normed, query_normed)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Rank by similarity descending
    ranked_indices = np.argsort(similarities)[::-1]
    results = []
    for rank, idx in enumerate(ranked_indices[:5], 1):
        results.append(
            {
                "rank": rank,
                "chunk_id": chunks[idx]["id"],
                "similarity": similarities[idx],
                "text_preview": chunks[idx]["text"][:120] + "...",
            }
        )
    return pd.DataFrame(results)


def results_to_df(results: dict, strategy_label: str) -> pd.DataFrame:
    """Convert ChromaDB query results into a tidy DataFrame."""
    return pd.DataFrame(
        {
            "strategy": strategy_label,
            "rank": range(1, len(results["ids"][0]) + 1),
            "distance": [round(d, 4) for d in results["distances"][0]],
            "text_preview": [doc[:120] + "..." for doc in results["documents"][0]],
        }
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_retrieval(
    ground_truth: list[dict],
    collection,
    query_embeddings_fn,
    k: int = 5,
) -> pd.DataFrame:
    """Run queries against a ChromaDB collection and compute retrieval metrics."""
    rows = []
    for entry in ground_truth:
        query = entry["query"]
        relevant = set(entry["relevant_ids"])

        q_vec = query_embeddings_fn(query)
        results = collection.query(
            query_embeddings=[q_vec],
            n_results=k,
        )
        retrieved_ids = results["ids"][0]

        hits_in_k = len(relevant & set(retrieved_ids))
        recall = hits_in_k / len(relevant) if relevant else 0.0
        precision = hits_in_k / k

        mrr = 0.0
        for rank, rid in enumerate(retrieved_ids, 1):
            if rid in relevant:
                mrr = 1.0 / rank
                break

        rows.append(
            {
                "query": query[:60] + "..." if len(query) > 60 else query,
                f"recall@{k}": recall,
                f"precision@{k}": precision,
                "mrr": mrr,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Copper-equivalent (CuEq) calculation
# ---------------------------------------------------------------------------

# Path to the cleaned World Bank commodity price CSV (nominal US$)
# Generated by scripts/extract_world_bank_prices.py from CMO-Historical-Data-Annual.xlsx
WORLD_BANK_PRICES_PATH = Path(__file__).resolve().parent / "data" / "raw" / "world_bank_commodity_prices.csv"

# Rolling window length for the price factor calculation (years)
CUEQ_ROLLING_WINDOW = 10


def load_commodity_prices(path: Path = WORLD_BANK_PRICES_PATH) -> pd.DataFrame:
    """Load the World Bank annual commodity price CSV.

    Args:
        path: Path to the CSV file with columns year, copper, nickel, etc.

    Returns:
        DataFrame indexed by year with one column per commodity (US$/unit).
    """
    df = pd.read_csv(path)
    df = df.set_index("year")
    return df


def compute_price_factor(
    commodity: str,
    year: int,
    prices_df: pd.DataFrame,
    window: int = CUEQ_ROLLING_WINDOW,
) -> float:
    """Compute the rolling-average price factor for a commodity vs copper.

    price_factor = avg_commodity_price / avg_copper_price
    over the preceding *window* years (inclusive of *year*).

    Args:
        commodity: Column name in the prices DataFrame (e.g. 'nickel', 'gold').
        year: Target year (end of the rolling window).
        prices_df: DataFrame from ``load_commodity_prices``.
        window: Number of years in the rolling average.

    Returns:
        Price factor (dimensionless ratio).

    Raises:
        ValueError: If commodity not found or insufficient data.
    """
    if commodity not in prices_df.columns:
        raise ValueError(
            f"Unknown commodity '{commodity}'. "
            f"Available: {sorted(prices_df.columns.tolist())}"
        )
    if "copper" not in prices_df.columns:
        raise ValueError("Copper column missing from price data.")

    start_year = year - window + 1
    mask = (prices_df.index >= start_year) & (prices_df.index <= year)
    window_df = prices_df.loc[mask, [commodity, "copper"]].dropna()

    if len(window_df) < window:
        logger.warning(
            "Only %d of %d years available for %s price factor (%d–%d)",
            len(window_df), window, commodity, start_year, year,
        )
    if window_df.empty:
        raise ValueError(
            f"No price data for '{commodity}' in range {start_year}–{year}."
        )

    avg_commodity = window_df[commodity].mean()
    avg_copper = window_df["copper"].mean()

    if avg_copper == 0:
        raise ValueError("Average copper price is zero — cannot compute factor.")

    return avg_commodity / avg_copper


def compute_copper_equivalent(
    sales_tonnes: float,
    commodity: str,
    year: int,
    prices_df: pd.DataFrame | None = None,
    window: int = CUEQ_ROLLING_WINDOW,
) -> dict[str, float]:
    """Calculate copper-equivalent volume for a commodity sale.

    CuEq = sales_tonnes × price_factor
    where price_factor = 10-year avg(commodity price) / 10-year avg(copper price).

    Args:
        sales_tonnes: Sales volume in metric tonnes.
        commodity: Commodity column name (e.g. 'nickel', 'gold', 'iron_ore').
        year: Report year (end of the rolling price window).
        prices_df: Pre-loaded prices DataFrame (loaded on demand if None).
        window: Rolling average window in years.

    Returns:
        Dict with keys:
            - ``sales_tonnes``: Input sales volume.
            - ``commodity``: Commodity name.
            - ``year``: Report year.
            - ``price_factor``: Computed ratio.
            - ``copper_equivalent_tonnes``: CuEq result.
    """
    if prices_df is None:
        prices_df = load_commodity_prices()

    price_factor = compute_price_factor(commodity, year, prices_df, window)
    cueq = sales_tonnes * price_factor

    return {
        "sales_tonnes": sales_tonnes,
        "commodity": commodity,
        "year": year,
        "price_factor": round(price_factor, 6),
        "copper_equivalent_tonnes": round(cueq, 2),
    }


# ---------------------------------------------------------------------------
# RetrievalConfig — shared configuration dataclass
# ---------------------------------------------------------------------------

# Default file-system paths
CHROMA_DIR = Path("data/interim/chromadb")
COLLECTION_NAME = "tpi_chunks"

# Embedding / retrieval defaults
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RETRIEVE_K = 50  # broad pool returned by bi-encoder
FINAL_K = 5  # final results after cross-encoder reranking
RRF_K = 60  # reciprocal rank fusion constant

# Generation defaults — validated in testing/generation_tests/distilgpt2_generation.ipynb
GEN_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
GEN_CONTEXT_WINDOW = 2048  # capped from 32k native to keep CPU generation fast
MAX_NEW_TOKENS = 256  # max tokens generated per answer

# Repetition penalty > 1.0 prevents degenerate token loops (e.g. "0.000...")
REPETITION_PENALTY = 1.2

# Cross-encoder model for reranking retrieved candidates
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Empirical overhead for Qwen chat template tokens (special role tags)
CHAT_TEMPLATE_OVERHEAD = 20


@dataclass
class RetrievalConfig:
    """Centralised configuration for retrieval and generation stages."""

    chroma_dir: Path = field(default_factory=lambda: CHROMA_DIR)
    collection_name: str = COLLECTION_NAME
    embedding_model_name: str = EMBEDDING_MODEL_NAME
    retrieve_k: int = RETRIEVE_K
    final_k: int = FINAL_K
    rrf_k: int = RRF_K
    gen_model: str = GEN_MODEL
    gen_context_window: int = GEN_CONTEXT_WINDOW
    max_new_tokens: int = MAX_NEW_TOKENS


# ---------------------------------------------------------------------------
# Query expansion
# ---------------------------------------------------------------------------

# Maps domain terms to alternative phrasings found in corporate disclosures.
# Each key is matched case-insensitively against the query; all alternatives
# are used as additional retrieval queries and fused via RRF.
QUERY_EXPANSIONS: dict[str, list[str]] = {
    "activity": [
        "production volume tonnes",
        "sales volume output tonnes",
        "copper equivalent production",
    ],
    "copper equivalent": [
        "CuEq production volume",
        "copper equivalent output tonnes",
        "total production volume",
    ],
}


def expand_query(query: str) -> list[str]:
    """Generate expanded query variants using domain-term mappings.

    Args:
        query: Original natural-language query.

    Returns:
        List of expanded queries (may be empty if no expansions match).
    """
    expansions: list[str] = []
    query_lower = query.lower()
    for term, alternatives in QUERY_EXPANSIONS.items():
        if term in query_lower:
            for alt in alternatives:
                expansions.append(query_lower.replace(term, alt))
    return expansions


# ---------------------------------------------------------------------------
# BM25 + RRF hybrid retrieval helpers
# ---------------------------------------------------------------------------

# Regex for BM25 tokenisation: word characters only
_WORD_RE = re.compile(r"\w+")


def tokenize_bm25(text: str) -> list[str]:
    """Lower-case word tokenisation for BM25.

    Args:
        text: Input text to tokenise.

    Returns:
        List of lower-cased word tokens.
    """
    return _WORD_RE.findall(text.lower())


def bm25_rank(
    query: str,
    doc_ids: list[str],
    doc_texts: list[str],
    k: int = RETRIEVE_K,
) -> list[str]:
    """Rank documents by BM25 score and return top-k IDs.

    Args:
        query: Natural-language query string.
        doc_ids: Document identifiers aligned with *doc_texts*.
        doc_texts: Document text strings to rank.
        k: Number of top-ranked IDs to return.

    Returns:
        Top-k document IDs ordered by descending BM25 score.
    """
    # Lazy import: rank_bm25 is only needed during retrieval and is heavy to load
    from rank_bm25 import BM25Okapi

    tokenised_docs = [tokenize_bm25(t) for t in doc_texts]
    bm25 = BM25Okapi(tokenised_docs)
    scores = bm25.get_scores(tokenize_bm25(query))
    ranked_idx = np.argsort(scores)[::-1][:k]
    return [doc_ids[i] for i in ranked_idx]


def rrf_fuse(
    ranked_lists: list[list[str]],
    rrf_k: int = RRF_K,
    k: int = RETRIEVE_K,
) -> list[str]:
    """Merge multiple ranked ID lists using Reciprocal Rank Fusion.

    Args:
        ranked_lists: Two or more lists of document IDs, each ordered by rank.
        rrf_k: Smoothing constant for RRF scoring.
        k: Number of top-fused IDs to return.

    Returns:
        Top-k document IDs ordered by descending RRF score.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, 1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
    sorted_ids = sorted(scores, key=scores.get, reverse=True)  # type: ignore[arg-type]
    return sorted_ids[:k]


def rerank_cross_encoder(
    query: str,
    doc_ids: list[str],
    doc_texts: list[str],
    k: int = FINAL_K,
) -> tuple[list[str], list[str]]:
    """Re-rank candidates with a cross-encoder and return top-k (ids, texts).

    Args:
        query: Natural-language query string.
        doc_ids: Candidate document IDs aligned with *doc_texts*.
        doc_texts: Candidate document text strings.
        k: Number of top-ranked results to return.

    Returns:
        Tuple of (top-k IDs, top-k texts), both ordered by descending
        cross-encoder relevance score.
    """
    # Lazy import: CrossEncoder loads a ~90 MB model; only needed for reranking
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(CROSS_ENCODER_MODEL)
    pairs = [[query, t] for t in doc_texts]
    scores = model.predict(pairs)
    ranked_idx = np.argsort(scores)[::-1][:k]
    return (
        [doc_ids[i] for i in ranked_idx],
        [doc_texts[i] for i in ranked_idx],
    )


def retrieve_chunks(
    query: str,
    config: RetrievalConfig | None = None,
) -> dict:
    """End-to-end hybrid retrieval: dense + BM25 → RRF → cross-encoder rerank.

    When the query matches a domain term in ``QUERY_EXPANSIONS``, additional
    dense searches are run with expanded phrasings and their results are
    fused via RRF alongside the original query results.

    Args:
        query: Natural-language query string.
        config: Pipeline configuration. Uses defaults when *None*.

    Returns:
        Dict with keys ``query``, ``ids``, ``texts``, ``metadatas``.
    """
    if config is None:
        config = RetrievalConfig()

    # --- load models and collection -----------------------------------------
    model = load_embedding_model(config.embedding_model_name)
    client = get_chroma_client(config.chroma_dir)
    collection = client.get_collection(
        name=config.collection_name,
        embedding_function=None,
    )

    # --- dense retrieval (original query) -----------------------------------
    q_vec = embed_texts(model, [query])[0]
    dense_results = collection.query(
        query_embeddings=[q_vec],
        n_results=config.retrieve_k,
        include=["documents", "metadatas"],
    )
    dense_ids: list[str] = dense_results["ids"][0]
    dense_texts: list[str] = dense_results["documents"][0]
    dense_meta: list[dict] = dense_results["metadatas"][0]

    # build lookup for later
    id_to_text = dict(zip(dense_ids, dense_texts))
    id_to_meta = dict(zip(dense_ids, dense_meta))

    # --- query expansion: additional dense retrieval -----------------------
    expanded_queries = expand_query(query)
    expansion_ranked_lists: list[list[str]] = []
    for eq in expanded_queries:
        eq_vec = embed_texts(model, [eq])[0]
        eq_results = collection.query(
            query_embeddings=[eq_vec],
            n_results=config.retrieve_k,
            include=["documents", "metadatas"],
        )
        eq_ids: list[str] = eq_results["ids"][0]
        eq_texts: list[str] = eq_results["documents"][0]
        eq_meta: list[dict] = eq_results["metadatas"][0]
        expansion_ranked_lists.append(eq_ids)
        # merge into lookups (original query results take precedence)
        for eid, etxt, emeta in zip(eq_ids, eq_texts, eq_meta):
            id_to_text.setdefault(eid, etxt)
            id_to_meta.setdefault(eid, emeta)
    if expanded_queries:
        logger.info(
            "Query expansion: %d extra queries for '%s'",
            len(expanded_queries),
            query,
        )

    # --- BM25 retrieval over the combined candidate pool -------------------
    all_candidate_ids = list(id_to_text.keys())
    all_candidate_texts = [id_to_text[i] for i in all_candidate_ids]
    bm25_ids = bm25_rank(
        query, all_candidate_ids, all_candidate_texts, k=config.retrieve_k
    )

    # --- RRF fusion (original dense + expanded dense + BM25) ---------------
    ranked_lists = [dense_ids] + expansion_ranked_lists + [bm25_ids]
    fused_ids = rrf_fuse(ranked_lists, rrf_k=config.rrf_k, k=config.retrieve_k)
    fused_texts = [id_to_text[i] for i in fused_ids]

    # --- cross-encoder reranking -------------------------------------------
    final_ids, final_texts = rerank_cross_encoder(
        query, fused_ids, fused_texts, k=config.final_k
    )
    final_meta = [id_to_meta[i] for i in final_ids]

    return {
        "query": query,
        "ids": final_ids,
        "texts": final_texts,
        "metadatas": final_meta,
    }


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

# System prompt — v2 stricter variant validated in generation notebook
SYSTEM_MESSAGE = (
    "You are a research assistant specialising in mining-sector carbon "
    "performance. Answer the question using ONLY the sources provided "
    "by the user.\n\n"
    "Domain context:\n"
    "- 'Activity' in the TPI Carbon Performance framework means total "
    "production volume, typically measured in tonnes of copper equivalent "
    "(tCuEq) for diversified miners.\n"
    "- Copper equivalent converts each commodity's output into an "
    "equivalent tonnage of copper using long-run commodity price ratios.\n"
    "- When asked about 'activity', look for production volumes, sales "
    "volumes, output tonnages, or copper equivalent figures in the "
    "sources.\n\n"
    "Rules:\n"
    "1. Quote all numbers, percentages, and years EXACTLY as they appear "
    "in the sources. Do not round or paraphrase.\n"
    "2. Cite the source number after each claim, e.g. [Source 1].\n"
    "3. If the sources mention a topic but do not give the exact figure, "
    "write: 'The sources mention [topic] but do not give the exact "
    "figure.'\n"
    "4. If the sources do not contain the answer at all, say: 'I cannot "
    "find this information in the provided sources.'\n"
    "5. Structure your answer as a numbered list with one claim per line."
)


def compute_token_budget(
    question: str,
    system_message: str,
    context_window: int,
    tokenizer: "object",
    max_new_tokens: int,
) -> int:
    """Compute the number of tokens available for retrieved context.

    Args:
        question: User question text.
        system_message: System prompt text.
        context_window: Maximum context length in tokens.
        tokenizer: A HuggingFace tokenizer with an ``encode`` method.
        max_new_tokens: Tokens reserved for generation output.

    Returns:
        Token budget available for retrieved-context chunks.
    """
    overhead = len(tokenizer.encode(system_message + question, add_special_tokens=True))  # type: ignore[union-attr]
    return context_window - overhead - max_new_tokens - CHAT_TEMPLATE_OVERHEAD


def trim_chunks_to_budget(
    texts: list[str],
    ids: list[str],
    budget: int,
    tokenizer: "object",
) -> tuple[list[str], list[str]]:
    """Keep the top chunks that fit within *budget* tokens.

    Args:
        texts: Chunk text strings in rank order.
        ids: Chunk IDs aligned with *texts*.
        budget: Maximum number of tokens to include.
        tokenizer: A HuggingFace tokenizer with an ``encode`` method.

    Returns:
        Tuple of (kept texts, kept IDs).
    """
    kept_texts: list[str] = []
    kept_ids: list[str] = []
    used = 0
    for text, cid in zip(texts, ids):
        toks = len(tokenizer.encode(text, add_special_tokens=False))  # type: ignore[union-attr]
        if used + toks > budget:
            break
        kept_texts.append(text)
        kept_ids.append(cid)
        used += toks
    return kept_texts, kept_ids


def build_prompt(
    question: str,
    context_texts: list[str],
    context_ids: list[str],
    tokenizer: "object",
) -> str:
    """Build a chat-template prompt with retrieved context.

    Uses ``apply_chat_template`` for Qwen2.5-0.5B-Instruct
    (system/user/assistant roles).

    Args:
        question: User question text.
        context_texts: Retrieved chunk texts.
        context_ids: Chunk IDs aligned with *context_texts*.
        tokenizer: A HuggingFace tokenizer with ``apply_chat_template``.

    Returns:
        Formatted prompt string ready for tokenisation.
    """
    context_block = "\n\n".join(
        f"[Source {i + 1}: {cid}]\n{text}"
        for i, (cid, text) in enumerate(zip(context_ids, context_texts))
    )
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": f"{context_block}\n\nQuestion: {question}"},
    ]
    return tokenizer.apply_chat_template(  # type: ignore[union-attr]
        messages, tokenize=False, add_generation_prompt=True
    )


def generate_answer(
    query: str,
    config: RetrievalConfig | None = None,
    retrieval_result: dict | None = None,
) -> dict:
    """Retrieve context and generate an answer with Qwen2.5-0.5B-Instruct.

    If *retrieval_result* is provided, skip retrieval and use those chunks.

    Args:
        query: Natural-language query string.
        config: Pipeline configuration. Uses defaults when *None*.
        retrieval_result: Pre-computed retrieval dict. When *None*,
            ``retrieve_chunks`` is called automatically.

    Returns:
        Dict with keys ``query``, ``answer``, ``sources``.
    """
    # Lazy import: transformers pulls in large model code; only needed for generation
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if config is None:
        config = RetrievalConfig()

    # --- retrieval ----------------------------------------------------------
    if retrieval_result is None:
        retrieval_result = retrieve_chunks(query, config)

    texts = retrieval_result["texts"]
    ids = retrieval_result["ids"]
    metadatas = retrieval_result["metadatas"]

    # --- load generation model ----------------------------------------------
    logger.info("Loading generation model %s", config.gen_model)
    tokenizer = AutoTokenizer.from_pretrained(config.gen_model)
    model = AutoModelForCausalLM.from_pretrained(config.gen_model)
    model.eval()

    # --- trim context to token budget --------------------------------------
    budget = compute_token_budget(
        query,
        SYSTEM_MESSAGE,
        config.gen_context_window,
        tokenizer,
        config.max_new_tokens,
    )
    trimmed_texts, trimmed_ids = trim_chunks_to_budget(texts, ids, budget, tokenizer)
    if not trimmed_texts:
        return {
            "query": query,
            "answer": "Context too long; no chunks fit the token budget.",
            "sources": [],
        }

    # --- build prompt and generate -----------------------------------------
    prompt = build_prompt(query, trimmed_texts, trimmed_ids, tokenizer)
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    logger.info(
        "Generating answer (%d input tokens, max_new=%d)",
        input_ids.shape[1],
        config.max_new_tokens,
    )

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=config.max_new_tokens,
            do_sample=False,
            repetition_penalty=REPETITION_PENALTY,
        )

    raw = tokenizer.decode(
        output_ids[0][input_ids.shape[1] :], skip_special_tokens=True
    )
    answer = raw.strip()

    # --- format sources ----------------------------------------------------
    sources = [
        {
            "id": cid,
            "source": meta.get("source", ""),
            "company": meta.get("company", ""),
        }
        for cid, meta in zip(
            trimmed_ids, [metadatas[ids.index(i)] for i in trimmed_ids]
        )
    ]

    return {"query": query, "answer": answer, "sources": sources}