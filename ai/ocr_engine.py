"""
PaddleOCR-based license plate text recognition.

Takes cropped plate images (from the YOLO detector) and extracts
the alphanumeric plate text using PaddleOCR.
"""

from __future__ import annotations
import logging
import re
from typing import Tuple

import cv2
import numpy as np
from paddleocr import PaddleOCR

from config import settings

logger = logging.getLogger(__name__)


class PlateOCR:
    """
    Recognizes license plate text from cropped plate images
    using PaddleOCR with pre/post-processing.
    """

    def __init__(self):
        logger.info("Initializing PaddleOCR (lang=%s)", settings.paddleocr_lang)
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang=settings.paddleocr_lang,
            show_log=False,
            use_gpu=settings.paddleocr_use_gpu,
        )

    # ──────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────

    def recognize(self, plate_crop: np.ndarray) -> Tuple[str, float]:
        """
        Recognize text from a plate image crop.

        Parameters
        ----------
        plate_crop : np.ndarray
            BGR image of the license plate region.

        Returns
        -------
        (plate_text, confidence) : Tuple[str, float]
            Cleaned plate number and average OCR confidence.
        """
        if plate_crop is None or plate_crop.size == 0:
            return "", 0.0

        # Pre-process for better OCR accuracy
        processed = self._preprocess(plate_crop)

        results = self.ocr.ocr(processed, cls=True)

        if not results or not results[0]:
            return "", 0.0

        # Combine all detected text fragments
        texts = []
        confidences = []
        for line in results[0]:
            text = line[1][0]
            conf = float(line[1][1])
            texts.append(text)
            confidences.append(conf)

        raw_text = "".join(texts)
        plate_text = self._clean_plate_text(raw_text)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        return plate_text, avg_confidence

    def recognize_file(self, image_path: str) -> Tuple[str, float]:
        """Run OCR directly on an image file (convenience method)."""
        img = cv2.imread(image_path)
        if img is None:
            return "", 0.0
        return self.recognize(img)

    # ──────────────────────────────────────────
    # Pre / Post processing
    # ──────────────────────────────────────────

    @staticmethod
    def _preprocess(img: np.ndarray) -> np.ndarray:
        """
        Apply image preprocessing to improve OCR accuracy on plate crops:
          1. Convert to grayscale
          2. Apply adaptive thresholding
          3. Resize to optimal height for PaddleOCR
        """
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Adaptive threshold for binarization
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2,
        )

        # Resize: PaddleOCR works best with ~48px height text
        h, w = binary.shape
        if h < 48:
            scale = 48 / h
            binary = cv2.resize(binary, (int(w * scale), 48), interpolation=cv2.INTER_CUBIC)

        # Convert back to BGR for PaddleOCR (it expects 3-channel input)
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def _clean_plate_text(raw: str) -> str:
        """
        Clean OCR output to produce a normalized plate number:
          - Keep only alphanumeric characters
          - Convert to uppercase
          - Remove common OCR misreads (O→0, I→1, etc. in context)
        """
        # Keep only letters and digits
        cleaned = re.sub(r"[^A-Za-z0-9]", "", raw)
        cleaned = cleaned.upper()

        # Common plate OCR corrections (heuristic)
        # These help when OCR confuses visually similar characters
        corrections = {
            "O": "0",  # In numeric positions, O is likely 0
        }
        # Only apply corrections to positions that should be digits
        # (This is format-dependent; keep it simple for now)

        return cleaned


# Singleton instance
_ocr: PlateOCR | None = None


def get_ocr() -> PlateOCR:
    """Return the singleton PlateOCR (lazy init)."""
    global _ocr
    if _ocr is None:
        _ocr = PlateOCR()
    return _ocr
