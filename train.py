# train.py
from ultralytics import YOLO
import wandb
import os

wandb.init(
    project="license-plate-detection",
    name="yolo-training",
    config={
        "model": "yolov8n.pt",
        "epochs": 30,
        "imgsz": 640,
        "batch": 8
    }
)

model = YOLO("yolov8n.pt")

results = model.train(
    data="dataset/data.yaml",
    epochs=30,
    imgsz=640,
    batch=8,
    device="cpu",
    workers=0,
    project="runs/train",
    name="license_plate",
    exist_ok=True,
    verbose=True
)

metrics = model.val()
print(f"✅ mAP50: {metrics.box.map50:.4f}")
print(f"✅ mAP50-95: {metrics.box.map:.4f}")

wandb.log({
    "mAP50": metrics.box.map50,
    "mAP50-95": metrics.box.map
})

wandb.finish()

print("Обучение завершено!")
print("Модель сохранена в runs/train/license_plate/weights/best.pt")
