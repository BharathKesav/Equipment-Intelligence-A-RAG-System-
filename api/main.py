# api/main.py

import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import time

from src.pipeline import run_pipeline


# ── Pydantic models — define request and response shapes ──

class QueryRequest(BaseModel):
    """What the client sends us."""
    question: str = Field(
        ...,                          # ... means required
        min_length=3,
        max_length=1000,
        description="The question to answer from the documents",
        example="What is the hydraulic pressure specification for CAT 320?"
    )

class Citation(BaseModel):
    """One source citation in the response."""
    number:        int
    paragraph_ref: str
    source:        str
    page:          int

class QueryResponse(BaseModel):
    """What we send back to the client."""
    answer:      str
    citations:   list[Citation]
    declined:    bool
    confidence:  float
    prompt_ver:  str
    duration_ms: float             # how long the query took


class HealthResponse(BaseModel):
    status:      str
    pipeline:    str
    chunks_indexed: str


# ── App startup and shutdown ──

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once when the server starts.
    We load all heavy models here so they're ready in memory
    before the first request arrives — not on the first request itself.
    """
    print("Starting RAG API server...")
    print("Models loading — this takes ~30 seconds on first run...")
    # run_pipeline imports trigger model loading at module level
    # (the _hybrid_retriever and _reranker globals in pipeline.py)
    print("RAG API ready to serve requests\n")
    yield
    # Code after yield runs on shutdown (cleanup if needed)
    print("RAG API shutting down...")


# ── Create the FastAPI app ──

app = FastAPI(
    title="Production RAG API",
    description=(
        "A hybrid retrieval RAG system with BGE reranking, "
        "scope guard, and paragraph-level citations."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Allow requests from any origin (needed if you build a separate frontend)
# In production you'd restrict this to your frontend's domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ──

@app.get("/health", response_model=HealthResponse)
def health_check():
    """
    Simple health check endpoint.
    Returns 200 if the server is running.
    Used by CI/CD and monitoring systems.
    """
    return HealthResponse(
        status="ok",
        pipeline="hybrid_rag_v1",
        chunks_indexed="check logs",
    )


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """
    Main RAG endpoint.
    Accepts a question, runs the full pipeline, returns answer + citations.
    """
    start_time = time.time()

    try:
        result = run_pipeline(request.question)
    except Exception as e:
        # If something unexpected crashes inside the pipeline,
        # return a 500 error instead of letting FastAPI show raw Python errors
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline error: {str(e)}"
        )

    duration_ms = (time.time() - start_time) * 1000

    return QueryResponse(
        answer      = result["answer"],
        citations   = result.get("citations", []),
        declined    = result["declined"],
        confidence  = result["confidence"],
        prompt_ver  = result.get("prompt_ver", "unknown"),
        duration_ms = round(duration_ms, 1),
    )