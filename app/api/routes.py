import logging

from fastapi import APIRouter, HTTPException, status

from app import model as embedder
from app.config import settings
from app.schemas import (
    EmbedRequest,
    EmbedResponse,
    HealthResponse,
    ReadyResponse,
    SingleEmbedding,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/embed", response_model=EmbedResponse)
async def embed(req: EmbedRequest) -> EmbedResponse:
    if len(req.texts) > settings.max_texts_per_request:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Too many texts: max {settings.max_texts_per_request}, got {len(req.texts)}",
        )
    if not embedder.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is still loading — retry after /ready returns 200",
        )

    logger.debug("Embedding %d text(s), input_type=%s", len(req.texts), req.input_type)
    try:
        dense_vecs, sparse_vecs = await embedder.embed(req.texts)
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))

    assert len(dense_vecs) == len(req.texts), "embed() returned wrong number of vectors"

    return EmbedResponse(
        model=settings.model_name,
        dense_dim=len(dense_vecs[0]),
        embeddings=[
            SingleEmbedding(dense=d, sparse=s)
            for d, s in zip(dense_vecs, sparse_vecs)
        ],
    )


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=embedder.is_loaded())


@router.get("/ready", response_model=ReadyResponse)
async def ready() -> ReadyResponse:
    if not embedder.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not yet loaded",
        )
    return ReadyResponse(status="ready")
