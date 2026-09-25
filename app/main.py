import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.redis_client import close_redis
from app.routers.websocket import router as ws_router
from app.services.dca_model_service import get_dca_model_runner
from app.tunnel import start_quick_tunnel, stop_quick_tunnel

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_dca_model_runner()
    tunnel = await start_quick_tunnel()
    try:
        yield
    finally:
        await stop_quick_tunnel(tunnel)
        await close_redis()


app = FastAPI(title="Groovo AI Server", version="0.1.0", lifespan=lifespan)
app.include_router(ws_router)


@app.get("/health")
async def health():
    return {"status": "ok"}
