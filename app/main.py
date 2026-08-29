import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.redis_client import close_redis
from app.routers.websocket import router as ws_router

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await close_redis()


app = FastAPI(title="Groovo AI Server", version="0.1.0", lifespan=lifespan)
app.include_router(ws_router)


@app.get("/health")
async def health():
    return {"status": "ok"}
