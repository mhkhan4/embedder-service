import asyncio
import logging

import torch
from FlagEmbedding import BGEM3FlagModel

from app.config import settings

logger = logging.getLogger(__name__)

_model: BGEM3FlagModel | None = None

# Lazy — constructed on first use so it binds to the running event loop (Python 3.10+).
_infer_semaphore: asyncio.Semaphore | None = None


def _get_semaphore() -> asyncio.Semaphore:
    global _infer_semaphore
    if _infer_semaphore is None:
        # Serialise concurrent encode() calls — CPU inference does not benefit from
        # parallelism; torch already uses multiple threads internally.
        _infer_semaphore = asyncio.Semaphore(settings.max_concurrent_inferences)
    return _infer_semaphore


def load_model() -> None:
    global _model
    if settings.torch_num_threads > 0:
        torch.set_num_threads(settings.torch_num_threads)
    logger.info("Loading %s (fp16=%s) on CPU…", settings.model_name, settings.use_fp16)
    _model = BGEM3FlagModel(
        settings.model_name,
        use_fp16=settings.use_fp16,
        devices="cpu",
    )
    logger.info("Model loaded.")


def unload_model() -> None:
    global _model
    _model = None


def is_loaded() -> bool:
    return _model is not None


async def embed(texts: list[str]) -> tuple[list[list[float]], list[dict[str, float]]]:
    """Return (dense_vectors, sparse_vectors) for the given texts.

    dense:  list of 1024-dim L2-normalised float lists
    sparse: list of {token_id_str: weight} dicts (BGE-M3 lexical_weights)
    """
    if _model is None:
        raise RuntimeError("Model not loaded — service is still warming up")

    def _run() -> tuple[list[list[float]], list[dict[str, float]]]:
        out = _model.encode(
            texts,
            batch_size=settings.model_batch_size,
            max_length=settings.max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense = [v.tolist() for v in out["dense_vecs"]]
        # Normalise keys to str so they survive JSON round-trips unambiguously.
        sparse = [
            {str(tid): float(w) for tid, w in lw.items()}
            for lw in out["lexical_weights"]
        ]
        return dense, sparse

    async with _get_semaphore():
        return await asyncio.to_thread(_run)
