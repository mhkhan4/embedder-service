import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import model as embedder
from app.api.routes import router
from app.config import settings

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "embedding-service starting — model=%s fp16=%s",
        settings.model_name,
        settings.use_fp16,
    )
    # Load off the event loop — model init is blocking CPU/IO work (~10-30 s)
    await asyncio.to_thread(embedder.load_model)
    logger.info("embedding-service ready")
    yield
    await asyncio.to_thread(embedder.unload_model)
    logger.info("embedding-service shut down")


app = FastAPI(title="embedding-service", lifespan=lifespan)
app.include_router(router)
