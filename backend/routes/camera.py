"""
Camera streaming and detection API routes.
Provides MJPEG stream and JSON detection polling endpoints.
"""

import logging
from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse, JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/camera", tags=["Camera"])


@router.get("/stream")
async def camera_stream(
    source: str = Query("demo", description="'demo', 'webcam', or file path"),
):
    """
    Stream annotated video frames as MJPEG.
    Shows real-time vehicle detection, tracking, plate OCR, and tax status.
    """
    from ai.camera import get_camera
    camera = get_camera(source=source)
    return StreamingResponse(
        camera.generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.get("/status")
async def camera_status():
    """Check which detection backends are available."""
    from ai.camera import (_YOLO_AVAILABLE, _OCR_AVAILABLE,
                           _TESSERACT_AVAILABLE)
    parts = []
    if _YOLO_AVAILABLE:
        parts.append("YOLO")
    else:
        parts.append("BG-Subtraction")
    if _OCR_AVAILABLE:
        parts.append("PaddleOCR")
    elif _TESSERACT_AVAILABLE:
        parts.append("Tesseract")
    else:
        parts.append("No-OCR")

    mode = " + ".join(parts)
    return {
        "yolo_available": _YOLO_AVAILABLE,
        "ocr_available": _OCR_AVAILABLE,
        "tesseract_available": _TESSERACT_AVAILABLE,
        "mode": mode,
    }


@router.get("/detections")
async def camera_detections():
    """
    Return current detection results as JSON for frontend polling.
    Includes tracked vehicles, plate OCR results, and tax status.
    """
    from ai.camera import get_camera, _camera
    if _camera is None:
        return JSONResponse(content={"detections": [], "frame": 0})

    # Enrich detections with tax status from database
    detections = list(_camera.latest_detections)

    # Look up tax status for confirmed plates (synchronous quick lookup)
    try:
        from backend.database import async_session_factory
        from backend import crud

        async with async_session_factory() as db:
            for det in detections:
                plate = det.get("plate_text", "")
                if plate and len(plate) >= 4:
                    vehicle = await crud.get_vehicle_by_plate(db, plate)
                    if vehicle:
                        det["tax_status"] = vehicle.tax_status.value
                        det["vehicle_found"] = True
                        det["owner_name"] = vehicle.owner_name
                    else:
                        det["tax_status"] = None
                        det["vehicle_found"] = False
    except Exception as e:
        logger.debug("Tax status lookup failed: %s", e)

    return JSONResponse(content={
        "detections": detections,
        "frame": _camera.frame_count,
        "source": _camera.source_name,
        "tracking_count": len(_camera.tracker.tracks),
    })
