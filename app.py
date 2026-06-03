"""FastAPI application for serving the RAG pipeline.

Run with:  uvicorn app:app --reload
"""

import logging
import os

# Windows DLL workaround (see utils.py docstring)
if os.name == "nt":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    import torch  # noqa: E402, F401

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from utils import RetrievalConfig, generate_answer, retrieve_chunks

logger = logging.getLogger(__name__)

app = FastAPI(
    title="TPI Carbon Performance RAG",
    description="Query carbon performance disclosures for Diversified Mining companies.",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    """Payload for the /query endpoint."""

    query: str = Field(
        ...,
        min_length=1,
        description="Natural-language question about carbon performance.",
    )
    final_k: int = Field(
        default=RetrievalConfig().final_k,
        ge=1,
        le=50,
        description="Number of source chunks.",
    )
    generate: bool = Field(
        default=True,
        description="If False, return only retrieved chunks (no LLM generation).",
    )


class SourceItem(BaseModel):
    """A single retrieved source chunk reference."""

    id: str
    source: str
    company: str


class QueryResponse(BaseModel):
    """Response payload for the /query endpoint."""

    query: str
    answer: str | None = None
    sources: list[SourceItem] = []
    texts: list[str] = []


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest) -> QueryResponse:
    """Answer a natural-language question using retrieved context.

    Args:
        req: Request body containing the query and options.

    Returns:
        QueryResponse with the answer, source chunks, and metadata.

    Raises:
        HTTPException: On retrieval or generation failure (500).
    """
    config = RetrievalConfig(final_k=req.final_k)

    try:
        retrieval_result = retrieve_chunks(req.query, config)
    except Exception as exc:
        logger.exception("Retrieval failed")
        raise HTTPException(status_code=500, detail=f"Retrieval error: {exc}") from exc

    if not req.generate:
        sources = [
            SourceItem(
                id=cid,
                source=meta.get("source", ""),
                company=meta.get("company", ""),
            )
            for cid, meta in zip(retrieval_result["ids"], retrieval_result["metadatas"])
        ]
        return QueryResponse(
            query=req.query,
            texts=retrieval_result["texts"],
            sources=sources,
        )

    try:
        gen_result = generate_answer(
            req.query, config=config, retrieval_result=retrieval_result
        )
    except Exception as exc:
        logger.exception("Generation failed")
        raise HTTPException(status_code=500, detail=f"Generation error: {exc}") from exc

    sources = [SourceItem(**s) for s in gen_result.get("sources", [])]
    return QueryResponse(
        query=gen_result["query"],
        answer=gen_result["answer"],
        sources=sources,
        texts=retrieval_result["texts"],
    )


@app.get("/health")
def health() -> dict:
    """Health check endpoint.

    Returns:
        Dict with ``status`` key.
    """
    return {"status": "ok"}