"""
TaxTrack - Central configuration module.
Reads settings from .env file or environment variables.
"""

from pathlib import Path
from pydantic_settings import BaseSettings


BASE_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    """Application settings loaded from environment / .env file."""

    # App
    app_name: str = "TaxTrack"
    app_version: str = "1.0.0"
    debug: bool = True

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Database
    database_url: str = f"sqlite+aiosqlite:///{BASE_DIR / 'taxtrack.db'}"

    # YOLO
    yolo_model_path: str = "yolov8n.pt"
    yolo_confidence_threshold: float = 0.45
    yolo_iou_threshold: float = 0.50

    # PaddleOCR
    paddleocr_lang: str = "en"
    paddleocr_use_gpu: bool = False

    # Tesseract (fallback OCR when PaddleOCR unavailable)
    tesseract_cmd: str = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    # Camera / Detection
    camera_video_path: str = "videos/traffic_demo.mp4"
    camera_process_every: int = 3  # Run detection every N frames
    camera_min_vehicle_area: float = 0.005  # Min fraction of frame for vehicle
    camera_ocr_min_confidence: float = 0.40  # Min OCR confidence to accept
    camera_tracking_max_distance: int = 120  # Max centroid distance for same track
    camera_ocr_votes_needed: int = 2  # Min matching OCR reads to confirm plate

    # Uploads
    upload_dir: str = "uploads"
    max_upload_size_mb: int = 10

    class Config:
        env_file = str(BASE_DIR / ".env")
        env_file_encoding = "utf-8"


settings = Settings()

# Ensure upload directory exists
(BASE_DIR / settings.upload_dir).mkdir(parents=True, exist_ok=True)
