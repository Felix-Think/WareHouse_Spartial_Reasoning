from ultralytics import YOLO  # type: ignore

# Load model segmentation
model = YOLO("runs/segment/train/weights/best.pt")

# model.train(data="data.yaml", imgsz=640, epochs=100, batch=16, device=0)

output = model(
    "datasets/valid/images/000502_png.rf.7f5840d02d128d64a16807dedcd24e71.jpg",
    save=True,
    conf=0.8,
    project=".",
    name="predict",
    boxes=False,
)
