from __future__ import annotations

import base64
from typing import Dict, List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results


def _decode_image(file_bytes: bytes) -> np.ndarray:
    """Decode uploaded bytes into a BGR image."""
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image data.")
    return img


def _encode_image_b64(img: np.ndarray) -> str:
    ok, buffer = cv2.imencode(".jpg", img)
    if not ok:
        raise ValueError("Could not encode annotated image.")
    b64 = base64.b64encode(buffer.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _draw_boxes(image: np.ndarray, result: Results) -> Tuple[np.ndarray, List[Dict]]:
    annotated = image.copy()
    names = result.names or {}
    boxes_out: List[Dict] = []
    if result.boxes is None:
        return annotated, boxes_out

    # simple deterministic palette
    palette = [
        (0, 127, 255),
        (120, 80, 220),
        (0, 200, 180),
        (255, 140, 0),
        (40, 180, 99),
        (231, 76, 60),
    ]

    for idx, box in enumerate(result.boxes, start=1):
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
        cls_id = int(box.cls.item()) if box.cls is not None else -1
        conf = float(box.conf.item()) if box.conf is not None else None
        cls_name = names.get(cls_id, str(cls_id))
        color = palette[cls_id % len(palette)] if cls_id >= 0 else palette[idx % len(palette)]

        cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        label = f"{idx}. {cls_name}"
        if conf is not None:
            label += f" {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(annotated, (int(x1), int(y1) - th - 8), (int(x1) + tw + 6, int(y1)), color, -1)
        cv2.putText(
            annotated,
            label,
            (int(x1) + 3, int(y1) - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        boxes_out.append(
            {
                "index": idx,
                "box": [x1, y1, x2, y2],
                "class_id": cls_id,
                "class_name": cls_name,
                "confidence": conf,
            }
        )

    return annotated, boxes_out


def run_detection(model: YOLO, file_bytes: bytes, conf: float) -> Dict:
    image = _decode_image(file_bytes)
    results = model.predict(source=image, conf=conf, verbose=False)
    result = results[0]
    annotated, boxes = _draw_boxes(image, result)
    return {
        "boxes": boxes,
        "image_size": {"width": int(image.shape[1]), "height": int(image.shape[0])},
        "annotated_image": _encode_image_b64(annotated),
    }
