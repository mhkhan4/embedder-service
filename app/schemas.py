from typing import Literal

from pydantic import BaseModel, Field


class EmbedRequest(BaseModel):
    texts: list[str] = Field(min_length=1)
    # BGE-M3 is symmetric — this field does NOT change the model call.
    # Retained for caller clarity, logging, and future instruction-prefix experiments.
    input_type: Literal["passage", "query"] = "passage"


class SingleEmbedding(BaseModel):
    dense: list[float]
    sparse: dict[str, float]


class EmbedResponse(BaseModel):
    model: str
    dense_dim: int
    embeddings: list[SingleEmbedding]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ReadyResponse(BaseModel):
    status: str
