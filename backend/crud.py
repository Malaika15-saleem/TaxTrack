"""
CRUD operations for vehicles, scan records, and tax records.
All functions are async and use SQLAlchemy 2.0 style queries.
"""

import datetime
from typing import List, Optional

from sqlalchemy import select, func, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.models import Vehicle, ScanRecord, TaxRecord, TaxStatus


# ──────────────────────────────────────────────
# Vehicle CRUD
# ──────────────────────────────────────────────

async def create_vehicle(db: AsyncSession, **kwargs) -> Vehicle:
    vehicle = Vehicle(**kwargs)
    db.add(vehicle)
    await db.flush()
    await db.refresh(vehicle)
    return vehicle


async def get_vehicle_by_id(db: AsyncSession, vehicle_id: int) -> Optional[Vehicle]:
    result = await db.execute(
        select(Vehicle).options(
            selectinload(Vehicle.scan_records),
            selectinload(Vehicle.tax_records),
        ).where(Vehicle.id == vehicle_id)
    )
    return result.scalars().first()


async def get_vehicle_by_plate(db: AsyncSession, plate: str) -> Optional[Vehicle]:
    result = await db.execute(
        select(Vehicle).where(Vehicle.plate_number == plate.upper())
    )
    return result.scalars().first()


async def get_all_vehicles(
    db: AsyncSession,
    skip: int = 0,
    limit: int = 50,
    status: Optional[str] = None,
    search: Optional[str] = None,
) -> List[Vehicle]:
    query = select(Vehicle).order_by(Vehicle.created_at.desc())

    if status:
        query = query.where(Vehicle.tax_status == status)
    if search:
        query = query.where(
            Vehicle.plate_number.ilike(f"%{search}%")
            | Vehicle.owner_name.ilike(f"%{search}%")
        )

    query = query.offset(skip).limit(limit)
    result = await db.execute(query)
    return result.scalars().all()


async def update_vehicle(db: AsyncSession, vehicle_id: int, **kwargs) -> Optional[Vehicle]:
    vehicle = await get_vehicle_by_id(db, vehicle_id)
    if not vehicle:
        return None
    for key, value in kwargs.items():
        if value is not None and hasattr(vehicle, key):
            setattr(vehicle, key, value)
    vehicle.updated_at = datetime.datetime.utcnow()
    await db.flush()
    await db.refresh(vehicle)
    return vehicle


async def delete_vehicle(db: AsyncSession, vehicle_id: int) -> bool:
    vehicle = await get_vehicle_by_id(db, vehicle_id)
    if not vehicle:
        return False
    await db.delete(vehicle)
    await db.flush()
    return True


# ──────────────────────────────────────────────
# Scan Record CRUD
# ──────────────────────────────────────────────

async def create_scan_record(db: AsyncSession, **kwargs) -> ScanRecord:
    record = ScanRecord(**kwargs)
    db.add(record)
    await db.flush()
    await db.refresh(record)
    return record


async def get_recent_scans(db: AsyncSession, limit: int = 10) -> List[ScanRecord]:
    result = await db.execute(
        select(ScanRecord)
        .options(selectinload(ScanRecord.vehicle))
        .order_by(ScanRecord.scanned_at.desc())
        .limit(limit)
    )
    return result.scalars().all()


async def get_scans_by_plate(db: AsyncSession, plate: str, limit: int = 20) -> List[ScanRecord]:
    result = await db.execute(
        select(ScanRecord)
        .where(ScanRecord.plate_number == plate.upper())
        .order_by(ScanRecord.scanned_at.desc())
        .limit(limit)
    )
    return result.scalars().all()


# ──────────────────────────────────────────────
# Tax Record CRUD
# ──────────────────────────────────────────────

async def create_tax_record(db: AsyncSession, **kwargs) -> TaxRecord:
    record = TaxRecord(**kwargs)
    db.add(record)
    await db.flush()
    await db.refresh(record)
    return record


async def get_tax_records_by_vehicle(db: AsyncSession, vehicle_id: int) -> List[TaxRecord]:
    result = await db.execute(
        select(TaxRecord)
        .where(TaxRecord.vehicle_id == vehicle_id)
        .order_by(TaxRecord.payment_date.desc())
    )
    return result.scalars().all()


# ──────────────────────────────────────────────
# Dashboard Statistics
# ──────────────────────────────────────────────

async def get_dashboard_stats(db: AsyncSession) -> dict:
    """Aggregate statistics for the dashboard."""
    total = await db.execute(select(func.count(Vehicle.id)))
    total_vehicles = total.scalar() or 0

    status_counts = {}
    for status in TaxStatus:
        result = await db.execute(
            select(func.count(Vehicle.id)).where(Vehicle.tax_status == status)
        )
        status_counts[status.value] = result.scalar() or 0

    scan_count = await db.execute(select(func.count(ScanRecord.id)))
    total_scans = scan_count.scalar() or 0

    return {
        "total_vehicles": total_vehicles,
        "tax_paid": status_counts.get("paid", 0),
        "tax_unpaid": status_counts.get("unpaid", 0),
        "tax_expired": status_counts.get("expired", 0),
        "tax_pending": status_counts.get("pending", 0),
        "total_scans": total_scans,
    }
