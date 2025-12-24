from ultralytics import YOLO  # type: ignore

# Load model segmentation
model = YOLO("weights/best.pt")

# model.train(data="data.yaml", imgsz=640, epochs=100, batch=16, device=0)

output = model(
    "datasets/images/train/078200.png",
    save=True,
    conf=0.8,
    project=".",
    name="predict",
    boxes=False,
)
