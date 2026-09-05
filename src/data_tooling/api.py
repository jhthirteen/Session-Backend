"""FastAPI contract for the NLP visualizer (Checkpoint 4).

Mount in the main backend app with:
    from src.data_tooling.api import router as nlp_router
    app.include_router(nlp_router)

Or run standalone for local testing:
    .venv/bin/uvicorn src.data_tooling.api:app --reload

Contract (React):
    POST /api/nlp/query  {query: "..."}  -> QueryResponse
    GET  /api/nlp/health -> {"status": "ok"}

Cross-repo frontend (Session-Frontend on another port) needs CORS, which
create_app() enables. Allowed origins come from FRONTEND_ORIGINS
(comma-separated, default covers Vite + CRA dev servers).
"""

import os
from typing import List, Optional

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .agent import run_query
from .models import QueryResponse


def frontend_origins() -> List[str]:
    raw = os.environ.get(
        "FRONTEND_ORIGINS", "http://localhost:5173,http://localhost:3000"
    )
    return [o.strip() for o in raw.split(",") if o.strip()]


class QueryRequest(BaseModel):
    query: str = Field(
        min_length=1, max_length=2000, description="Natural-language NBA question."
    )
    model: Optional[str] = Field(
        default=None,
        description="Groq model override, e.g. 'llama-3.3-70b-versatile'.",
    )


router = APIRouter(prefix="/api/nlp", tags=["nlp"])


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.post("/query", response_model=QueryResponse)
def post_query(body: QueryRequest) -> QueryResponse:
    query = body.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="Query must not be empty.")
    try:
        return run_query(query, model=body.model)
    except HTTPException:
        raise
    except Exception as e:  # unexpected — surface as 500, never fake stats
        raise HTTPException(status_code=500, detail=f"NLP query failed: {e}")


def create_app() -> FastAPI:
    app = FastAPI(title="Session NLP Visualizer")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=frontend_origins(),
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "Authorization"],
    )
    app.include_router(router)
    return app


# Standalone app for uvicorn / TestClient.
app = create_app()
