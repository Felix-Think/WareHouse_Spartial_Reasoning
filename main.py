from ultralytics import YOLO

# Load model segmentation
model = YOLO("yolov12s-seg.pt")

model.train(
    data="data.yaml", imgsz=640, epochs=100, batch=16, device=0  # <-- scale 0.5
)
