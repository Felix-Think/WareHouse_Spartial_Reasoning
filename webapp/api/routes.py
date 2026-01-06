from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from ..dependencies.model import get_config, get_model
from ..services.detector import run_detection

router = APIRouter(prefix="/api")


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.post("/detect")
async def detect_image(
    file: UploadFile = File(...),
    model=Depends(get_model),
    config=Depends(get_config),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file.")

    try:
        result = run_detection(model, content, conf=config.conf_threshold)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Failed to process image: {exc}") from exc

    return result
