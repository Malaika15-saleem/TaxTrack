"""
Pydantic schemas for request validation and response serialization.
"""

from __future__ import annotations
import datetime
from typing import Optional, List
from pydantic import BaseModel, Field


# ──────────────────────────────────────────────
# Vehicle Schemas
# ──────────────────────────────────────────────

class VehicleBase(BaseModel):
    plate_number: str = Field(..., min_length=3, max_length=20, examples=["AB1234CD"])
    owner_name: str = Field(..., min_length=1, max_length=150, examples=["John Doe"])
    vehicle_type: str = Field(default="car", examples=["car", "truck", "motorcycle"])
    vehicle_brand: Optional[str] = Field(None, max_length=100, examples=["Toyota"])
    vehicle_color: Optional[str] = Field(None, max_length=50, examples=["White"])
    tax_status: str = Field(default="pending", examples=["paid", "unpaid", "expired", "pending"])
    tax_expiry_date: Optional[datetime.datetime] = None
    notes: Optional[str] = None


class VehicleCreate(VehicleBase):
    """Schema for creating a new vehicle."""
    pass


class VehicleUpdate(BaseModel):
    """Schema for partial vehicle updates."""
    owner_name: Optional[str] = Field(None, max_length=150)
    vehicle_type: Optional[str] = None
    vehicle_brand: Optional[str] = None
    vehicle_color: Optional[str] = None
    tax_status: Optional[str] = None
    tax_expiry_date: Optional[datetime.datetime] = None
    is_active: Optional[bool] = None
    notes: Optional[str] = None


class VehicleResponse(VehicleBase):
    """Schema returned by API endpoints."""
    id: int
    registration_date: datetime.datetime
    is_active: bool
    created_at: datetime.datetime
    updated_at: datetime.datetime

    model_config = {"from_attributes": True}


# ──────────────────────────────────────────────
# Tax Record Schemas
# ──────────────────────────────────────────────

class TaxRecordBase(BaseModel):
    vehicle_id: int
    amount: float = Field(..., gt=0, examples=[150.00])
    expiry_date: datetime.datetime
    payment_method: Optional[str] = Field(None, examples=["bank_transfer", "cash", "card"])
    receipt_number: Optional[str] = None


class TaxRecordCreate(TaxRecordBase):
    pass


class TaxRecordResponse(TaxRecordBase):
    id: int
    payment_date: datetime.datetime
    created_at: datetime.datetime

    model_config = {"from_attributes": True}


# ──────────────────────────────────────────────
# Scan / Detection Schemas
# ──────────────────────────────────────────────

class DetectionBox(BaseModel):
    """A single bounding box from YOLO detection."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    label: str


class OCRResult(BaseModel):
    """OCR result for a detected plate region."""
    text: str
    confidence: float


class ScanResponse(BaseModel):
    """Full response from a detection + OCR scan."""
    plate_number: str
    detection_confidence: float
    ocr_confidence: float
    detection_boxes: List[DetectionBox] = []
    ocr_text: str
    tax_status: Optional[str] = None
    vehicle_found: bool = False
    vehicle: Optional[VehicleResponse] = None
    image_path: Optional[str] = None
    scanned_at: datetime.datetime


# ──────────────────────────────────────────────
# Dashboard / Stats
# ──────────────────────────────────────────────

class DashboardStats(BaseModel):
    total_vehicles: int
    tax_paid: int
    tax_unpaid: int
    tax_expired: int
    tax_pending: int
    total_scans: int
    recent_scans: List[ScanResponse] = []


# ──────────────────────────────────────────────
# Generic
# ──────────────────────────────────────────────

class MessageResponse(BaseModel):
    message: str
    detail: Optional[str] = None
