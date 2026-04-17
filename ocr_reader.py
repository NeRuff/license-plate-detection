import cv2
import numpy as np
from typing import List, Dict, Optional
import easyocr
from loguru import logger

class LicensePlateOCR:
    def __init__(self, languages: list = None):
        self.languages = languages or ['en', 'ru']
        self.reader = easyocr.Reader(self.languages, gpu=False, verbose=False)
        logger.info(f"OCR initialized with languages: {self.languages}")
    
    def read_plate(self, frame: np.ndarray, bbox: List[int]) -> Optional[str]:
        x1, y1, x2, y2 = bbox
        plate_roi = frame[y1:y2, x1:x2]
        if plate_roi.size == 0:
            return None
        plate_roi = self._preprocess_image(plate_roi)
        results = self.reader.readtext(plate_roi, detail=0, paragraph=False)
        if results:
            import re
            text = ' '.join(results)
            text = re.sub(r'[^A-Za-z0-9]', '', text).upper()
            return text if len(text) >= 2 else None
        return None
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        gray = cv2.equalizeHist(gray)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
    
    def draw_text(self, frame: np.ndarray, bbox: List[int], text: str) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        if text:
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - 28), (x1 + text_w + 10, y1), (255, 100, 0), -1)
            cv2.putText(frame, text, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return frame
