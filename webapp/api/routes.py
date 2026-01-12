from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from ..dependencies.distance import get_distance_config, get_distance_model
from ..dependencies.model import get_config, get_model
from ..services.distance import predict_distance_from_question
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


@router.post("/distance")
async def distance_between_objects(
    question: str = Form(...),
    image: UploadFile = File(...),
    depth: UploadFile = File(...),
    model=Depends(get_model),
    config=Depends(get_config),
    distance_model=Depends(get_distance_model),
    distance_config=Depends(get_distance_config),
):
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")
    if not depth.content_type or not depth.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload a depth image file.")

    image_bytes = await image.read()
    depth_bytes = await depth.read()
    if not image_bytes or not depth_bytes:
        raise HTTPException(status_code=400, detail="Empty image or depth file.")

    try:
        detection = run_detection(model, image_bytes, conf=config.conf_threshold)
        result = predict_distance_from_question(
            question=question,
            boxes=detection["boxes"],
            image_size=detection["image_size"],
            depth_bytes=depth_bytes,
            model=distance_model,
            fovx_deg=distance_config.fovx_deg,
            fovy_deg=distance_config.fovy_deg,
            depth_max=distance_config.depth_max,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Failed to estimate distance: {exc}") from exc

    return {
        "question": question,
        "indices": [result["index1"], result["index2"]],
        "distance": result["distance"],
        "boxes": detection["boxes"],
        "image_size": detection["image_size"],
        "annotated_image": detection["annotated_image"],
    }
