from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .api.routes import router as api_router


def create_app() -> FastAPI:
    app = FastAPI(title="Warehouse Distance Detector", version="0.1.0")
    app.include_router(api_router)

    frontend_dir = Path(__file__).parent / "frontend"
    index_file = frontend_dir / "index.html"

    if frontend_dir.exists():
        app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

    @app.get("/", include_in_schema=False)
    async def root():
        if not index_file.exists():
            raise HTTPException(status_code=404, detail="Frontend not found.")
        return FileResponse(index_file)

    @app.get("/config", include_in_schema=False)
    async def config():
        # Handy for quick debugging of frontend/backend connectivity
        return JSONResponse({"status": "ok"})

    return app


app = create_app()
