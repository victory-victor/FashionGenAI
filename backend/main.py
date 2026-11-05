"""FastAPI entry point:  uvicorn main:app --reload   (run from the backend/ folder)"""
import asyncio
import os
import re
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

import config
from api.routes import router

app = FastAPI(title="FashionGen AI", version="1.0.0",
              description="Text-to-Conditional-GAN Bridge: LLM parsing -> cGAN generation -> CLIP evaluation")


@app.middleware("http")
async def clean_path_middleware(request: Request, call_next):
    # Normalize duplicate slashes in the path (e.g. //api/health -> /api/health)
    path = request.scope.get("path", "")
    if "//" in path:
        request.scope["path"] = re.sub(r"/+", "/", path)
    return await call_next(request)


app.add_middleware(
    CORSMiddleware,
    allow_origins=config.FRONTEND_ORIGINS if config.FRONTEND_ORIGINS else ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)
app.mount("/files/samples", StaticFiles(directory=config.SAMPLES_DIR), name="samples")


@app.on_event("startup")
async def start_keep_alive():
    """Keeps the service warm on Render free tier by pinging itself before the 15m idle sleep."""
    render_url = os.getenv("RENDER_EXTERNAL_URL") or os.getenv("BACKEND_URL")
    if not render_url:
        return

    clean_url = render_url.rstrip("/") + "/api/health"
    async def pinger():
        await asyncio.sleep(300)       # wait 5 min after first boot
        async with httpx.AsyncClient(timeout=15.0) as client:
            while True:
                try:
                    await client.get(clean_url)
                except Exception:
                    pass
                await asyncio.sleep(600)  # ping every 10 min (Render sleeps after 15 min idle)

    asyncio.create_task(pinger())


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    msg = "; ".join(f"{'.'.join(str(p) for p in e['loc'][1:])}: {e['msg']}" for e in exc.errors())
    return JSONResponse(status_code=422, content={"detail": f"Invalid input - {msg}"})


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    print("[error]", type(exc).__name__, exc)           # logs stay server-side
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})


# Root route: redirect to the UI at /app/ or API docs if frontend not mounted
_frontend = Path(__file__).resolve().parent / "frontend"
if not _frontend.is_dir():
    _frontend = Path(__file__).resolve().parent.parent / "frontend"

@app.get("/", include_in_schema=False)
def root():
    if config.SERVE_FRONTEND and _frontend.is_dir():
        return RedirectResponse(url="/app/")
    return {"message": "FashionGen AI API", "docs": "/docs", "health": "/api/health"}


if config.SERVE_FRONTEND and _frontend.is_dir():
    app.mount("/app", StaticFiles(directory=_frontend, html=True), name="frontend")
