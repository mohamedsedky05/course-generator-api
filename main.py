import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from config import settings
from routers.generate import router as generate_router
from utils.rate_limit import limiter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("course_generator")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — pre-loading Whisper model...")
    try:
        from services.transcriber import load_whisper_model
        load_whisper_model()
        logger.info("Whisper model loaded successfully.")
    except Exception as e:
        logger.warning(f"Could not pre-load Whisper model: {e}")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="AI Course Generator",
    description="Generate structured educational content from text, files, or YouTube videos.",
    version="1.0.0",
    lifespan=lifespan,
)

# Rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# GZip compression (applied to responses >= 1 KB)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# CORS — origins come from .env ALLOWED_ORIGINS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    logger.info(f"→ {request.method} {request.url.path} | ip={request.client.host}")
    response = await call_next(request)
    elapsed = round(time.time() - start, 3)
    logger.info(f"← {response.status_code} ({elapsed}s)")
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "error_code": "INTERNAL_ERROR",
            "message": "An unexpected error occurred.",
            "ar_message": "حدث خطأ غير متوقع. يرجى المحاولة مرة أخرى.",
        },
    )


app.include_router(generate_router)


@app.get("/")
async def root():
    return {"message": "AI Course Generator API is running. See /docs for usage."}
