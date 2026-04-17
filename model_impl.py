from ultralytics import YOLO
import numpy as np
from typing import List, Dict
from loguru import logger

class My_LicensePlate_Model:
    def __init__(self, model_path: str = "runs/detect/runs/train/license_plate/weights/best.pt"):
        self.model = YOLO(model_path)
        self.confidence_threshold = 0.25
        logger.info(f"Model loaded from {model_path}")
    
    def detect_plates(self, frame: np.ndarray) -> List[Dict]:
        results = self.model(frame, verbose=False)[0]
        plates = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            confidence = float(box.conf[0])
            if confidence >= self.confidence_threshold:
                plates.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": confidence
                })
        logger.debug(f"Found {len(plates)} plates")
        return plates
    
    def draw_boxes(self, frame: np.ndarray, plates: List[Dict]) -> np.ndarray:
        import cv2
        for plate in plates:
            x1, y1, x2, y2 = plate["bbox"]
            conf = plate["confidence"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{conf:.2f}", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame
