"""
SQLAlchemy ORM models for the TaxTrack database.

Schema overview:
  vehicles        - Registered vehicles and their tax status
  scan_records    - Every detection/OCR scan event
  tax_records     - Tax payment history per vehicle
"""

import datetime
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Boolean, ForeignKey, Text, Enum as SAEnum,
)
from sqlalchemy.orm import relationship
from backend.database import Base
import enum


# ──────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────

class TaxStatus(str, enum.Enum):
    PAID = "paid"
    UNPAID = "unpaid"
    EXPIRED = "expired"
    PENDING = "pending"


class VehicleType(str, enum.Enum):
    CAR = "car"
    TRUCK = "truck"
    MOTORCYCLE = "motorcycle"
    BUS = "bus"
    VAN = "van"
    OTHER = "other"


# ──────────────────────────────────────────────
# Models
# ──────────────────────────────────────────────

class Vehicle(Base):
    """A registered vehicle in the tax system."""
    __tablename__ = "vehicles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    plate_number = Column(String(20), unique=True, nullable=False, index=True)
    owner_name = Column(String(150), nullable=False)
    vehicle_type = Column(SAEnum(VehicleType), default=VehicleType.CAR, nullable=False)
    vehicle_brand = Column(String(100), nullable=True)
    vehicle_color = Column(String(50), nullable=True)
    registration_date = Column(DateTime, default=datetime.datetime.utcnow)
    tax_status = Column(SAEnum(TaxStatus), default=TaxStatus.PENDING, nullable=False)
    tax_expiry_date = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    # Relationships
    scan_records = relationship("ScanRecord", back_populates="vehicle", cascade="all, delete-orphan")
    tax_records = relationship("TaxRecord", back_populates="vehicle", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Vehicle plate={self.plate_number!r} status={self.tax_status}>"


class ScanRecord(Base):
    """A single detection + OCR scan event."""
    __tablename__ = "scan_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    plate_number = Column(String(20), nullable=False, index=True)
    image_path = Column(String(500), nullable=True)
    detection_confidence = Column(Float, nullable=True)
    ocr_confidence = Column(Float, nullable=True)
    vehicle_id = Column(Integer, ForeignKey("vehicles.id"), nullable=True)
    tax_status_matched = Column(SAEnum(TaxStatus), nullable=True)
    scanned_at = Column(DateTime, default=datetime.datetime.utcnow)

    # Relationships
    vehicle = relationship("Vehicle", back_populates="scan_records")

    def __repr__(self):
        return f"<ScanRecord plate={self.plate_number!r} conf={self.detection_confidence}>"


class TaxRecord(Base):
    """Tax payment history for a vehicle."""
    __tablename__ = "tax_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    vehicle_id = Column(Integer, ForeignKey("vehicles.id"), nullable=False)
    amount = Column(Float, nullable=False)
    payment_date = Column(DateTime, default=datetime.datetime.utcnow)
    expiry_date = Column(DateTime, nullable=False)
    payment_method = Column(String(50), nullable=True)
    receipt_number = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # Relationships
    vehicle = relationship("Vehicle", back_populates="tax_records")

    def __repr__(self):
        return f"<TaxRecord vehicle_id={self.vehicle_id} amount={self.amount}>"
