"""
Vehicle management API routes.
CRUD endpoints for vehicles, tax records, and dashboard statistics.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database import get_db
from backend import crud
from backend.schemas import (
    VehicleCreate,
    VehicleUpdate,
    VehicleResponse,
    TaxRecordCreate,
    TaxRecordResponse,
    DashboardStats,
    MessageResponse,
)

router = APIRouter(prefix="/api/vehicles", tags=["Vehicles"])


# ──────────────────────────────────────────────
# Vehicle Endpoints
# ──────────────────────────────────────────────

@router.post("/", response_model=VehicleResponse, status_code=201)
async def create_vehicle(data: VehicleCreate, db: AsyncSession = Depends(get_db)):
    """Register a new vehicle in the system."""
    existing = await crud.get_vehicle_by_plate(db, data.plate_number)
    if existing:
        raise HTTPException(status_code=409, detail=f"Vehicle with plate '{data.plate_number}' already exists")

    vehicle = await crud.create_vehicle(
        db,
        plate_number=data.plate_number.upper(),
        owner_name=data.owner_name,
        vehicle_type=data.vehicle_type,
        vehicle_brand=data.vehicle_brand,
        vehicle_color=data.vehicle_color,
        tax_status=data.tax_status,
        tax_expiry_date=data.tax_expiry_date,
        notes=data.notes,
    )
    return vehicle


@router.get("/", response_model=List[VehicleResponse])
async def list_vehicles(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    status: Optional[str] = Query(None, description="Filter by tax status"),
    search: Optional[str] = Query(None, description="Search plate or owner name"),
    db: AsyncSession = Depends(get_db),
):
    """List all vehicles with optional filtering and search."""
    vehicles = await crud.get_all_vehicles(db, skip=skip, limit=limit, status=status, search=search)
    return vehicles


@router.get("/{vehicle_id}", response_model=VehicleResponse)
async def get_vehicle(vehicle_id: int, db: AsyncSession = Depends(get_db)):
    """Get a single vehicle by ID."""
    vehicle = await crud.get_vehicle_by_id(db, vehicle_id)
    if not vehicle:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    return vehicle


@router.get("/plate/{plate_number}", response_model=VehicleResponse)
async def get_vehicle_by_plate(plate_number: str, db: AsyncSession = Depends(get_db)):
    """Look up a vehicle by its plate number."""
    vehicle = await crud.get_vehicle_by_plate(db, plate_number)
    if not vehicle:
        raise HTTPException(status_code=404, detail=f"No vehicle found with plate '{plate_number}'")
    return vehicle


@router.put("/{vehicle_id}", response_model=VehicleResponse)
async def update_vehicle(vehicle_id: int, data: VehicleUpdate, db: AsyncSession = Depends(get_db)):
    """Update a vehicle's information."""
    update_data = data.model_dump(exclude_unset=True)
    vehicle = await crud.update_vehicle(db, vehicle_id, **update_data)
    if not vehicle:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    return vehicle


@router.delete("/{vehicle_id}", response_model=MessageResponse)
async def delete_vehicle(vehicle_id: int, db: AsyncSession = Depends(get_db)):
    """Delete a vehicle and all associated records."""
    deleted = await crud.delete_vehicle(db, vehicle_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    return MessageResponse(message="Vehicle deleted successfully")


# ──────────────────────────────────────────────
# Tax Record Endpoints
# ──────────────────────────────────────────────

tax_router = APIRouter(prefix="/api/tax", tags=["Tax Records"])


@tax_router.post("/pay", response_model=TaxRecordResponse, status_code=201)
async def record_tax_payment(data: TaxRecordCreate, db: AsyncSession = Depends(get_db)):
    """Record a tax payment for a vehicle and update its status to 'paid'."""
    vehicle = await crud.get_vehicle_by_id(db, data.vehicle_id)
    if not vehicle:
        raise HTTPException(status_code=404, detail="Vehicle not found")

    record = await crud.create_tax_record(
        db,
        vehicle_id=data.vehicle_id,
        amount=data.amount,
        expiry_date=data.expiry_date,
        payment_method=data.payment_method,
        receipt_number=data.receipt_number,
    )

    # Update vehicle tax status to paid
    await crud.update_vehicle(db, data.vehicle_id, tax_status="paid", tax_expiry_date=data.expiry_date)

    return record


@tax_router.get("/history/{vehicle_id}", response_model=List[TaxRecordResponse])
async def get_tax_history(vehicle_id: int, db: AsyncSession = Depends(get_db)):
    """Get tax payment history for a vehicle."""
    records = await crud.get_tax_records_by_vehicle(db, vehicle_id)
    return records


# ──────────────────────────────────────────────
# Dashboard Stats Endpoint
# ──────────────────────────────────────────────

stats_router = APIRouter(prefix="/api", tags=["Dashboard"])


@stats_router.get("/stats", response_model=DashboardStats)
async def get_stats(db: AsyncSession = Depends(get_db)):
    """Get aggregated dashboard statistics."""
    stats = await crud.get_dashboard_stats(db)
    recent = await crud.get_recent_scans(db, limit=5)

    from backend.schemas import ScanResponse as SR
    recent_scans = []
    for s in recent:
        recent_scans.append(SR(
            plate_number=s.plate_number,
            detection_confidence=s.detection_confidence or 0,
            ocr_confidence=s.ocr_confidence or 0,
            ocr_text=s.plate_number,
            tax_status=s.tax_status_matched.value if s.tax_status_matched else None,
            vehicle_found=s.vehicle_id is not None,
            scanned_at=s.scanned_at,
        ))

    return DashboardStats(
        total_vehicles=stats["total_vehicles"],
        tax_paid=stats["tax_paid"],
        tax_unpaid=stats["tax_unpaid"],
        tax_expired=stats["tax_expired"],
        tax_pending=stats["tax_pending"],
        total_scans=stats["total_scans"],
        recent_scans=recent_scans,
    )
