"""
YOLO-based vehicle and license plate detector.

Uses Ultralytics YOLOv8 to detect vehicles in images, then crops
the license plate region for downstream OCR processing.

NOTE: The default yolov8n.pt model detects vehicles but NOT plates
directly. For production, fine-tune YOLOv8 on a license-plate dataset
(e.g., Roboflow "License Plate Recognition" datasets) and set the
path via YOLO_MODEL_PATH in your .env file.

This module falls back to a heuristic plate-region crop when no
dedicated plate model is available.
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from config import settings

logger = logging.getLogger(__name__)


# COCO class IDs relevant to vehicles
VEHICLE_CLASS_IDS = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


class PlateDetector:
    """
    Detects vehicles (and optionally license plates) using YOLOv8,
    then extracts plate-region crops for OCR.
    """

    def __init__(self):
        model_path = settings.yolo_model_path
        logger.info("Loading YOLO model: %s", model_path)
        self.model = YOLO(model_path)
        self.conf_threshold = settings.yolo_confidence_threshold
        self.iou_threshold = settings.yolo_iou_threshold
        self._has_plate_class = self._check_plate_class()

    def _check_plate_class(self) -> bool:
        """Return True if the loaded model has a 'license-plate' class."""
        names = self.model.names  # dict {id: name}
        return any("plate" in name.lower() for name in names.values())

    # ──────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────

    def detect(self, image_path: str | Path) -> dict:
        """
        Run detection on an image file.

        Returns
        -------
        dict with keys:
            vehicles   : list of (label, confidence, box)
            plates     : list of (confidence, box)
            plate_crops: list of np.ndarray (BGR crops for OCR)
            annotated  : np.ndarray (image with boxes drawn)
        """
        image_path = Path(image_path)
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")

        results = self.model(
            img,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )

        vehicles: list = []
        plates: list = []
        plate_crops: list = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                cls_name = result.names.get(cls_id, str(cls_id))

                # License plate class from fine-tuned model
                if "plate" in cls_name.lower():
                    plates.append((conf, (x1, y1, x2, y2)))
                    plate_crops.append(self._crop_region(img, x1, y1, x2, y2))

                # Standard vehicle detection
                elif cls_id in VEHICLE_CLASS_IDS:
                    vehicles.append((VEHICLE_CLASS_IDS[cls_id], conf, (x1, y1, x2, y2)))

        # If no dedicated plate model, estimate plate region from vehicle boxes
        if not plate_crops and vehicles:
            plate_crops, plates = self._estimate_plate_regions(img, vehicles)

        annotated = self._draw_annotations(img.copy(), vehicles, plates)

        return {
            "vehicles": vehicles,
            "plates": plates,
            "plate_crops": plate_crops,
            "annotated": annotated,
        }

    # ──────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────

    @staticmethod
    def _crop_region(img: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
        """Crop a bounding box region from the image with boundary clamping."""
        h, w = img.shape[:2]
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        return img[y1:y2, x1:x2].copy()

    @staticmethod
    def _estimate_plate_regions(
        img: np.ndarray,
        vehicles: list,
    ) -> Tuple[list, list]:
        """
        Heuristic: estimate license plate location from the bottom-center
        of each detected vehicle bounding box.
        """
        plate_crops = []
        plates = []
        h, w = img.shape[:2]

        for label, conf, (x1, y1, x2, y2) in vehicles:
            bw = x2 - x1
            bh = y2 - y1

            # Plate is typically in the lower 25-45% of a car, centered horizontally
            px1 = int(x1 + bw * 0.25)
            px2 = int(x1 + bw * 0.75)
            py1 = int(y1 + bh * 0.55)
            py2 = int(y1 + bh * 0.85)

            # Clamp to image boundaries
            px1, py1 = max(0, px1), max(0, py1)
            px2, py2 = min(w, px2), min(h, py2)

            if px2 > px1 and py2 > py1:
                crop = img[py1:py2, px1:px2].copy()
                if crop.size > 0:
                    plate_crops.append(crop)
                    plates.append((conf * 0.7, (px1, py1, px2, py2)))

        return plate_crops, plates

    @staticmethod
    def _draw_annotations(
        img: np.ndarray,
        vehicles: list,
        plates: list,
    ) -> np.ndarray:
        """Draw bounding boxes and labels on the image."""
        for label, conf, (x1, y1, x2, y2) in vehicles:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                img, f"{label} {conf:.2f}", (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,
            )
        for conf, (x1, y1, x2, y2) in plates:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                img, f"plate {conf:.2f}", (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2,
            )
        return img


# Singleton instance
_detector: PlateDetector | None = None


def get_detector() -> PlateDetector:
    """Return the singleton PlateDetector (lazy init)."""
    global _detector
    if _detector is None:
        _detector = PlateDetector()
    return _detector
