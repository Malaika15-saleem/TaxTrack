"""
Detection and OCR API routes.
Handles image upload, YOLO detection, PaddleOCR recognition,
and tax status lookup for detected plates.
"""

import datetime
import logging
from pathlib import Path

import cv2
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings, BASE_DIR
from backend.database import get_db
from backend import crud
from backend.schemas import ScanResponse, DetectionBox, MessageResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/detect", tags=["Detection"])


@router.post("/scan", response_model=ScanResponse)
async def scan_image(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """
    Upload a vehicle image and run the full detection + OCR pipeline:
      1. Save uploaded image
      2. Run YOLO vehicle/plate detection
      3. Extract plate region and run PaddleOCR
      4. Look up the recognized plate in the database
      5. Return plate number + tax status
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are accepted")

    # Save uploaded file
    upload_dir = BASE_DIR / settings.upload_dir
    upload_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{timestamp}_{file.filename}"
    file_path = upload_dir / filename

    content = await file.read()
    max_size = settings.max_upload_size_mb * 1024 * 1024
    if len(content) > max_size:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size is {settings.max_upload_size_mb} MB",
        )

    with open(file_path, "wb") as f:
        f.write(content)

    # Run detection (lazy-load AI models on first call)
    try:
        from ai.detector import get_detector
        from ai.ocr_engine import get_ocr

        detector = get_detector()
        ocr = get_ocr()

        result = detector.detect(file_path)

        # Extract detection boxes for the response
        detection_boxes = []
        for label, conf, (x1, y1, x2, y2) in result["vehicles"]:
            detection_boxes.append(DetectionBox(
                x1=x1, y1=y1, x2=x2, y2=y2, confidence=conf, label=label,
            ))
        for conf, (x1, y1, x2, y2) in result["plates"]:
            detection_boxes.append(DetectionBox(
                x1=x1, y1=y1, x2=x2, y2=y2, confidence=conf, label="license_plate",
            ))

        # Run OCR on plate crops
        plate_text = ""
        ocr_confidence = 0.0
        detection_confidence = 0.0

        if result["plate_crops"]:
            plate_text, ocr_confidence = ocr.recognize(result["plate_crops"][0])
            if result["plates"]:
                detection_confidence = result["plates"][0][0]
        elif result["vehicles"]:
            detection_confidence = result["vehicles"][0][1]

        # If OCR failed, try direct OCR on the full image
        if not plate_text:
            plate_text, ocr_confidence = ocr.recognize_file(str(file_path))

        # Save annotated image
        annotated_path = upload_dir / f"annotated_{filename}"
        cv2.imwrite(str(annotated_path), result["annotated"])

        # Look up plate in database
        vehicle = None
        vehicle_found = False
        tax_status = None

        if plate_text:
            vehicle = await crud.get_vehicle_by_plate(db, plate_text)
            if vehicle:
                vehicle_found = True
                tax_status = vehicle.tax_status.value

            # Save scan record
            await crud.create_scan_record(
                db,
                plate_number=plate_text,
                image_path=str(file_path),
                detection_confidence=detection_confidence,
                ocr_confidence=ocr_confidence,
                vehicle_id=vehicle.id if vehicle else None,
                tax_status_matched=vehicle.tax_status if vehicle else None,
            )

        return ScanResponse(
            plate_number=plate_text or "UNRECOGNIZED",
            detection_confidence=round(detection_confidence, 4),
            ocr_confidence=round(ocr_confidence, 4),
            detection_boxes=detection_boxes,
            ocr_text=plate_text or "Could not read plate",
            tax_status=tax_status,
            vehicle_found=vehicle_found,
            vehicle=vehicle,
            image_path=str(annotated_path),
            scanned_at=datetime.datetime.utcnow(),
        )

    except Exception as e:
        logger.exception("Detection pipeline failed")
        raise HTTPException(status_code=500, detail=f"Detection pipeline error: {str(e)}")


@router.post("/scan/plate-text", response_model=ScanResponse)
async def lookup_plate_text(
    plate: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Look up a plate number directly (without image detection).
    Useful for manual plate entry.
    """
    plate = plate.strip().upper()
    if not plate:
        raise HTTPException(status_code=400, detail="Plate number cannot be empty")

    vehicle = await crud.get_vehicle_by_plate(db, plate)
    vehicle_found = vehicle is not None
    tax_status = vehicle.tax_status.value if vehicle else None

    # Record the manual lookup
    await crud.create_scan_record(
        db,
        plate_number=plate,
        detection_confidence=1.0,
        ocr_confidence=1.0,
        vehicle_id=vehicle.id if vehicle else None,
        tax_status_matched=vehicle.tax_status if vehicle else None,
    )

    return ScanResponse(
        plate_number=plate,
        detection_confidence=1.0,
        ocr_confidence=1.0,
        ocr_text=plate,
        tax_status=tax_status,
        vehicle_found=vehicle_found,
        vehicle=vehicle,
        scanned_at=datetime.datetime.utcnow(),
    )
