import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
 
from app.config import settings
from app.database import Base, engine
from app import models  # noqa: F401  (import so tables are registered on Base.metadata)
from app.model_loader import get_recommender
from app.routers import auth, chat, interactions, products, recommendations
from app.seed import seed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    rec = get_recommender()  # load model once at startup
    logger.info("Recommender loaded: %s", rec.is_loaded)
    result = seed()
    logger.info("Seed: %s", result)
    yield


app = FastAPI(
    title="Amazon Electronics Recommendation Chatbot API",
    version="0.2.0",
    lifespan=lifespan,
)

origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

for r in (auth.router, products.router, interactions.router, recommendations.router, chat.router):
    app.include_router(r)


@app.get("/health")
def health():
    rec = get_recommender()
    return {
        "status": "ok",
        "model_loaded": rec.is_loaded,
        "catalog_size": rec.catalog_size,
        "alpha": rec.alpha,
    }


@app.get("/")
def root():
    return {"app": "Amazon Electronics Recommendation Chatbot API", "docs": "/docs"}
