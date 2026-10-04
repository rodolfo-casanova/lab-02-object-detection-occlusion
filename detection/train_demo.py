"""Optional (GPU recommended): short retraining with targeted data augmentation.

  docker compose --profile gpu run --rm training

Augmentations: random erasing, mosaic and copy-paste, which simulate occlusion and background variety.
"""
import os

from common.utils import cache_path, download, results_dir
from ultralytics import YOLO

ASSETS = "https://github.com/ultralytics/assets/releases/download/v8.3.0/"

if __name__ == "__main__":
    model = YOLO(str(download(ASSETS + "yolov8n.pt", cache_path("models", "yolov8n.pt"))))
    model.train(data="coco128.yaml", epochs=int(os.getenv("EPOCHS", "3")), imgsz=640,
                erasing=0.4, mosaic=1.0, copy_paste=0.3, device=os.getenv("DEVICE", "cpu"),
                project=str(results_dir("detection") / "train"), name="targeted_augmentation", exist_ok=True)
