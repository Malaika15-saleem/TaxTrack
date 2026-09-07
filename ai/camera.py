"""
Live camera / video feed processor for real-time vehicle detection,
tracking, plate detection, and OCR.

Pipeline:
  Video → Vehicle Detection → Vehicle Tracking → Plate Region Crop
  → OCR (multi-frame voting) → Tax Database Check → Annotated Stream

Supports:
  - Full AI: YOLO detection + PaddleOCR (when packages installed)
  - Fallback: OpenCV BG-subtraction vehicle detection + Tesseract OCR

Sources:
  - "demo"  → videos/traffic_demo.mp4 (real traffic video)
  - "webcam"→ laptop camera 0 via DirectShow
"""

from __future__ import annotations

import logging
import math
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional, Generator

import cv2
import numpy as np

from config import settings, BASE_DIR

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# AI availability flags
# ──────────────────────────────────────────────

_YOLO_AVAILABLE = False
_OCR_AVAILABLE = False
_TESSERACT_AVAILABLE = False

try:
    from ai.detector import get_detector
    _YOLO_AVAILABLE = True
    logger.info("YOLO detector available")
except ImportError:
    logger.warning("YOLO not installed — using OpenCV BG-subtraction fallback")

try:
    from ai.ocr_engine import get_ocr
    _OCR_AVAILABLE = True
    logger.info("PaddleOCR available")
except ImportError:
    logger.warning("PaddleOCR not installed")

try:
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd
    _ver = pytesseract.get_tesseract_version()
    _TESSERACT_AVAILABLE = True
    logger.info("Tesseract %s available as OCR fallback", _ver)
except Exception:
    logger.warning("Tesseract not available")


# ──────────────────────────────────────────────
# Background subtractor (OpenCV headless compatible)
# ──────────────────────────────────────────────

_bg_subtractor = None


def _get_bg_subtractor():
    global _bg_subtractor
    if _bg_subtractor is None:
        _bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=40, detectShadows=False,
        )
    return _bg_subtractor


def _reset_bg_subtractor():
    global _bg_subtractor
    _bg_subtractor = None


# ──────────────────────────────────────────────
# Simple centroid-based vehicle tracker
# ──────────────────────────────────────────────

class _VehicleTrack:
    """Represents a tracked vehicle across frames."""
    __slots__ = ("track_id", "cx", "cy", "box", "label", "confidence",
                 "frames_seen", "plate_text", "plate_confidence",
                 "plate_votes", "frames_since_plate")

    def __init__(self, track_id, cx, cy, box, label, confidence):
        self.track_id = track_id
        self.cx = cx
        self.cy = cy
        self.box = box  # (x1, y1, x2, y2)
        self.label = label
        self.confidence = confidence
        self.frames_seen = 0
        self.plate_text = ""  # confirmed plate
        self.plate_confidence = 0.0
        self.plate_votes: dict[str, int] = defaultdict(int)
        self.frames_since_plate = 0


class CentroidTracker:
    """Simple centroid tracker for vehicle bounding boxes."""

    def __init__(self, max_distance: int = 120, max_lost: int = 15):
        self.max_distance = max_distance
        self.max_lost = max_lost
        self._next_id = 1
        self.tracks: dict[int, _VehicleTrack] = {}

    def reset(self):
        self.tracks.clear()
        self._next_id = 1

    def update(self, detections: list[dict]) -> dict[int, _VehicleTrack]:
        """
        Update tracks with new detections.
        detections: list of {label, confidence, box: (x1,y1,x2,y2)}
        Returns dict of active tracks.
        """
        if not detections:
            # Mark all tracks as lost
            lost_ids = []
            for tid, track in self.tracks.items():
                track.frames_since_plate += 1
                if track.frames_since_plate > self.max_lost:
                    lost_ids.append(tid)
            for tid in lost_ids:
                del self.tracks[tid]
            return self.tracks

        # Compute centroids for new detections
        centroids = []
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            centroids.append((cx, cy, det))

        if not self.tracks:
            for cx, cy, det in centroids:
                self._register(cx, cy, det)
            return self.tracks

        track_ids = list(self.tracks.keys())
        track_centroids = [
            (self.tracks[tid].cx, self.tracks[tid].cy) for tid in track_ids
        ]

        # Distance matrix
        D = np.zeros((len(centroids), len(track_centroids)))
        for i, (cx, cy, _) in enumerate(centroids):
            for j, (tcx, tcy) in enumerate(track_centroids):
                D[i][j] = math.dist((cx, cy), (tcx, tcy))

        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if D[row, col] > self.max_distance:
                continue

            tid = track_ids[col]
            track = self.tracks[tid]
            cx, cy, det = centroids[row]
            track.cx = cx
            track.cy = cy
            track.box = det["box"]
            track.label = det["label"]
            track.confidence = det["confidence"]
            track.frames_seen += 1
            track.frames_since_plate = 0
            used_rows.add(row)
            used_cols.add(col)

        # Unmatched existing tracks
        for col in range(len(track_centroids)):
            if col not in used_cols:
                tid = track_ids[col]
                self.tracks[tid].frames_since_plate += 1
                if self.tracks[tid].frames_since_plate > self.max_lost:
                    del self.tracks[tid]

        # Unmatched new detections → new tracks
        for row in range(len(centroids)):
            if row not in used_rows:
                cx, cy, det = centroids[row]
                self._register(cx, cy, det)

        return self.tracks

    def _register(self, cx, cy, det):
        tid = self._next_id
        self._next_id += 1
        self.tracks[tid] = _VehicleTrack(tid, cx, cy, det["box"],
                                         det["label"], det["confidence"])


# ──────────────────────────────────────────────
# Tesseract OCR helper
# ──────────────────────────────────────────────

def _tesseract_ocr(crop: np.ndarray) -> tuple[str, float]:
    """Run Tesseract OCR on a plate crop. Returns (text, confidence)."""
    if not _TESSERACT_AVAILABLE or crop is None or crop.size == 0:
        return "", 0.0

    try:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # Resize small crops for better OCR
        h, w = gray.shape
        if h < 40:
            scale = 48 / h
            gray = cv2.resize(gray, (int(w * scale), 48),
                              interpolation=cv2.INTER_CUBIC)

        # Adaptive threshold
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2,
        )

        # Tesseract with PSM 7 (single text line) for plates
        custom_config = r"--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        data = pytesseract.image_to_data(binary, config=custom_config,
                                         output_type=pytesseract.Output.DICT)

        texts = []
        confs = []
        for i, txt in enumerate(data["text"]):
            txt = txt.strip()
            if txt:
                conf = int(data["conf"][i]) / 100.0
                if conf > 0:
                    texts.append(txt)
                    confs.append(conf)

        if not texts:
            return "", 0.0

        raw = "".join(texts).upper()
        cleaned = re.sub(r"[^A-Z0-9]", "", raw)
        avg_conf = sum(confs) / len(confs) if confs else 0.0

        return cleaned, avg_conf

    except Exception as e:
        logger.debug("Tesseract OCR error: %s", e)
        return "", 0.0


def _paddle_ocr(crop: np.ndarray) -> tuple[str, float]:
    """Run PaddleOCR on a plate crop."""
    if not _OCR_AVAILABLE or crop is None or crop.size == 0:
        return "", 0.0
    try:
        ocr = get_ocr()
        return ocr.recognize(crop)
    except Exception as e:
        logger.debug("PaddleOCR error: %s", e)
        return "", 0.0


def _run_ocr(crop: np.ndarray) -> tuple[str, float]:
    """Run the best available OCR on a plate crop."""
    if _OCR_AVAILABLE:
        text, conf = _paddle_ocr(crop)
        if text and conf >= settings.camera_ocr_min_confidence:
            return text, conf

    if _TESSERACT_AVAILABLE:
        return _tesseract_ocr(crop)

    return "", 0.0


# ──────────────────────────────────────────────
# Camera Processor
# ──────────────────────────────────────────────

class CameraProcessor:
    """
    Processes video frames for real-time vehicle detection, tracking,
    plate detection, and OCR. Yields annotated JPEG frames for MJPEG.
    """

    def __init__(self, source: str = "demo", process_every: int | None = None):
        self.source_name = source
        self.process_every = process_every or settings.camera_process_every
        self.frame_count = 0
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_webcam = (source == "webcam")

        # Tracking
        self.tracker = CentroidTracker(
            max_distance=settings.camera_tracking_max_distance,
        )

        # Latest detection results (for JSON polling)
        self.latest_detections: list[dict] = []

        # Resolve video path
        self.video_path = self._resolve_source(source)
        logger.info("Camera source: %s → %s (webcam=%s)",
                     source, self.video_path, self.is_webcam)

    def _resolve_source(self, source: str) -> str | int:
        if source == "webcam":
            return 0
        if source in ("demo", "dummy"):
            # Prefer new path, fallback to old
            new_path = BASE_DIR / settings.camera_video_path
            if new_path.exists():
                return str(new_path)
            old_path = BASE_DIR / "sample_data" / "dummy_traffic.mp4"
            if old_path.exists():
                return str(old_path)
            raise FileNotFoundError(
                f"Traffic video not found. Expected at {new_path}")
        return source  # direct file path

    def open(self):
        if self.is_webcam:
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        else:
            self.cap = cv2.VideoCapture(str(self.video_path))

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.video_path}")

        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info("Video opened: %dx%d @ %.1f fps, %d frames (webcam=%s)",
                     w, h, fps, frames, self.is_webcam)

    def close(self):
        if self.cap:
            self.cap.release()
            self.cap = None

    # ──────────────────────────────────────────
    # MJPEG stream
    # ──────────────────────────────────────────

    def generate_frames(self) -> Generator[bytes, None, None]:
        if not self.cap or not self.cap.isOpened():
            self.open()

        while True:
            ret, frame = self.cap.read()

            if not ret:
                if self.is_webcam:
                    time.sleep(0.01)
                    continue
                else:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.tracker.reset()
                    _reset_bg_subtractor()
                    continue

            self.frame_count += 1

            if self.frame_count % self.process_every == 0:
                self._process_frame(frame)

            annotated = self._annotate_frame(frame)

            ok, buffer = cv2.imencode(
                ".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + buffer.tobytes()
                + b"\r\n"
            )

            time.sleep(1 / 25)

    # ──────────────────────────────────────────
    # Detection pipeline
    # ──────────────────────────────────────────

    def _process_frame(self, frame: np.ndarray):
        """Full pipeline: detect → track → plate crop → OCR → vote."""
        # Step 1: Detect vehicles
        detections = self._detect_vehicles(frame)

        # Step 2: Update tracker
        active_tracks = self.tracker.update(detections)

        # Step 3: For each tracked vehicle, crop plate region and run OCR
        for tid, track in active_tracks.items():
            # Only attempt OCR every few frames per track (save CPU)
            if track.frames_seen % 3 != 0:
                continue

            plate_crop = self._extract_plate_region(frame, track.box)
            if plate_crop is None or plate_crop.size == 0:
                continue

            text, conf = _run_ocr(plate_crop)

            if text and len(text) >= 4 and conf >= settings.camera_ocr_min_confidence:
                track.plate_votes[text] += 1
                logger.debug("Track %d: OCR '%s' conf=%.2f (votes=%d)",
                             tid, text, conf, track.plate_votes[text])

                # Confirm plate when enough matching votes
                if track.plate_votes[text] >= settings.camera_ocr_votes_needed:
                    if track.plate_text != text:
                        logger.info("Track %d: plate confirmed '%s' "
                                    "(votes=%d, conf=%.2f)",
                                    tid, text, track.plate_votes[text], conf)
                    track.plate_text = text
                    track.plate_confidence = conf

        # Step 4: Build latest detections list (for JSON polling)
        self.latest_detections = []
        for tid, track in active_tracks.items():
            entry = {
                "track_id": tid,
                "label": track.label,
                "confidence": round(track.confidence, 3),
                "box": [round(v, 1) for v in track.box],
                "plate_text": track.plate_text or "",
                "plate_confidence": round(track.plate_confidence, 3),
                "frames_seen": track.frames_seen,
                "tax_status": None,  # filled by route via DB lookup
            }
            self.latest_detections.append(entry)

    def _detect_vehicles(self, frame: np.ndarray) -> list[dict]:
        """Detect vehicles using YOLO or BG-subtraction fallback."""
        if _YOLO_AVAILABLE:
            return self._detect_yolo(frame)
        return self._detect_bg_subtraction(frame)

    def _detect_yolo(self, frame: np.ndarray) -> list[dict]:
        """Use YOLO for vehicle detection."""
        try:
            import tempfile, os
            tmp = os.path.join(tempfile.gettempdir(), "taxtrack_frame.jpg")
            cv2.imwrite(tmp, frame)
            detector = get_detector()
            result = detector.detect(tmp)
            os.unlink(tmp)

            detections = []
            for label, conf, box in result["vehicles"]:
                detections.append({
                    "label": label,
                    "confidence": conf,
                    "box": tuple(box),
                })
            return detections
        except Exception as e:
            logger.error("YOLO detection error: %s", e)
            return self._detect_bg_subtraction(frame)

    def _detect_bg_subtraction(self, frame: np.ndarray) -> list[dict]:
        """OpenCV BG-subtraction + contour detection for vehicles."""
        subtractor = _get_bg_subtractor()
        fg_mask = subtractor.apply(frame)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel,
                                   iterations=2)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel,
                                   iterations=1)

        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        h, w = frame.shape[:2]
        min_area = (h * w) * settings.camera_min_vehicle_area

        valid = [(cv2.contourArea(c), c) for c in contours
                 if cv2.contourArea(c) > min_area]
        valid.sort(key=lambda x: x[0], reverse=True)

        detections = []
        for area, contour in valid[:8]:
            x, y, cw, ch = cv2.boundingRect(contour)
            aspect = cw / max(ch, 1)
            if aspect < 0.3 or aspect > 5.0:
                continue

            conf = min(0.95, 0.5 + (area / (h * w)) * 10)
            detections.append({
                "label": "vehicle",
                "confidence": round(conf, 2),
                "box": (float(x), float(y),
                        float(x + cw), float(y + ch)),
            })

        return detections

    @staticmethod
    def _extract_plate_region(frame: np.ndarray,
                              box: tuple) -> Optional[np.ndarray]:
        """Crop the estimated plate region from a vehicle bounding box."""
        x1, y1, x2, y2 = [int(v) for v in box]
        bw = x2 - x1
        bh = y2 - y1

        # Plate is typically in the lower-center of the vehicle
        px1 = int(x1 + bw * 0.20)
        px2 = int(x1 + bw * 0.80)
        py1 = int(y1 + bh * 0.55)
        py2 = int(y1 + bh * 0.90)

        h, w = frame.shape[:2]
        px1, py1 = max(0, px1), max(0, py1)
        px2, py2 = min(w, px2), min(h, py2)

        if px2 <= px1 or py2 <= py1:
            return None

        crop = frame[py1:py2, px1:px2]
        return crop if crop.size > 0 else None

    # ──────────────────────────────────────────
    # Frame annotation (debug overlays)
    # ──────────────────────────────────────────

    def _annotate_frame(self, frame: np.ndarray) -> np.ndarray:
        """Draw all debug overlays on the frame."""
        h, w = frame.shape[:2]
        active_tracks = self.tracker.tracks

        for tid, track in active_tracks.items():
            x1, y1, x2, y2 = [int(v) for v in track.box]

            # ── Vehicle bounding box (green) ──
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # ── Tracking ID + label ──
            id_label = f"ID:{tid} {track.label}"
            cv2.rectangle(frame, (x1, max(y1 - 30, 0)),
                          (x1 + len(id_label) * 8 + 8, max(y1 - 2, 18)),
                          (0, 0, 0), -1)
            cv2.putText(frame, id_label, (x1 + 4, max(y1 - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # ── Plate bounding box (red) ──
            bw = x2 - x1
            bh = y2 - y1
            px1 = int(x1 + bw * 0.20)
            px2 = int(x1 + bw * 0.80)
            py1 = int(y1 + bh * 0.55)
            py2 = int(y1 + bh * 0.90)
            px1, py1 = max(0, px1), max(0, py1)
            px2, py2 = min(w, px2), min(h, py2)

            if px2 > px1 and py2 > py1:
                cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 0, 255), 2)

            # ── OCR result + confidence ──
            if track.plate_text:
                plate_label = (f"{track.plate_text} "
                               f"({track.plate_confidence:.0%})")
                cv2.rectangle(frame, (px1, max(py1 - 28, 0)),
                              (px1 + len(plate_label) * 8 + 8,
                               max(py1 - 2, 16)),
                              (0, 0, 0), -1)
                cv2.putText(frame, plate_label,
                            (px1 + 4, max(py1 - 8, 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (0, 255, 255), 1)

                # ── Tax status badge ──
                status = track.plate_text  # will be looked up via DB
                status_color = (128, 128, 128)  # gray = unknown
                status_text = "UNKNOWN"
                # We check latest_detections which may have tax_status
                for det in self.latest_detections:
                    if det["track_id"] == tid and det.get("tax_status"):
                        status_text = det["tax_status"].upper()
                        if status_text == "PAID":
                            status_color = (0, 255, 0)
                        elif status_text == "UNPAID":
                            status_color = (0, 0, 255)
                        elif status_text == "EXPIRED":
                            status_color = (0, 165, 255)
                        break

                cv2.rectangle(frame, (x1, y2 + 2),
                              (x1 + len(status_text) * 9 + 10, y2 + 22),
                              (0, 0, 0), -1)
                cv2.putText(frame, status_text, (x1 + 4, y2 + 17),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            status_color, 1)
            else:
                # Show "Plate Not Read" if enough frames seen
                if track.frames_seen >= 5:
                    msg = "Plate Not Read"
                    cv2.rectangle(frame, (px1, max(py1 - 28, 0)),
                                  (px1 + len(msg) * 8 + 8,
                                   max(py1 - 2, 16)),
                                  (0, 0, 0), -1)
                    cv2.putText(frame, msg, (px1 + 4, max(py1 - 8, 14)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                (0, 0, 255), 1)

        # ── HUD overlay ──
        mode_parts = []
        if _YOLO_AVAILABLE:
            mode_parts.append("YOLO")
        else:
            mode_parts.append("BG-Subtraction")
        if _OCR_AVAILABLE:
            mode_parts.append("PaddleOCR")
        elif _TESSERACT_AVAILABLE:
            mode_parts.append("Tesseract")
        else:
            mode_parts.append("No-OCR")

        mode_text = " + ".join(mode_parts)
        cv2.rectangle(frame, (5, 5), (360, 35), (0, 0, 0), -1)
        cv2.putText(frame, f"DETECTION: {mode_text}", (12, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)

        # Frame counter
        frame_text = f"Frame: {self.frame_count}"
        cv2.rectangle(frame, (w - 150, 5), (w - 5, 35), (0, 0, 0), -1)
        cv2.putText(frame, frame_text, (w - 142, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        # Tracks count
        track_text = f"Tracking: {len(active_tracks)} vehicles"
        cv2.rectangle(frame, (5, 38), (260, 62), (0, 0, 0), -1)
        cv2.putText(frame, track_text, (12, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        # ── Bottom: confirmed plates list ──
        confirmed = [(tid, t) for tid, t in active_tracks.items()
                     if t.plate_text]
        if confirmed:
            y_off = h - 10
            for tid, track in reversed(confirmed[-4:]):
                votes = dict(track.plate_votes)
                text = (f">> ID:{tid} {track.plate_text}  "
                        f"conf:{track.plate_confidence:.0%}  "
                        f"votes:{votes.get(track.plate_text, 0)}")
                cv2.rectangle(frame, (5, y_off - 18),
                              (450, y_off + 2), (0, 0, 0), -1)
                cv2.putText(frame, text, (10, y_off),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                            (0, 255, 255), 1)
                y_off -= 22

        return frame


# ──────────────────────────────────────────────
# Singleton manager
# ──────────────────────────────────────────────

_camera: Optional[CameraProcessor] = None
_camera_source: Optional[str] = None


def get_camera(source: str = "demo") -> CameraProcessor:
    """Get or create the camera processor. Resets when source changes."""
    global _camera, _camera_source
    if _camera is not None and _camera_source != source:
        logger.info("Source changed: %s → %s, resetting camera",
                     _camera_source, source)
        _camera.close()
        _camera = None
    if _camera is None:
        _camera = CameraProcessor(source=source)
        _camera_source = source
    return _camera


def reset_camera():
    global _camera
    if _camera:
        _camera.close()
    _camera = None
