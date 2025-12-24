from ultralytics import YOLO

model = YOLO("yolov8s-seg.pt")

results = model.train(
    data=" datasets/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    device=0,
    workers=8,
)


output = model(
    "datasets/valid/images/000502_png.rf.7f5840d02d128d64a16807dedcd24e71.jpg",
    save=True,
    project="outputs",
    name="val_predict",
    exist_ok=True,
)
