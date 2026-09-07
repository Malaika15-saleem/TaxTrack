"""
Database seed script - populates the database with sample vehicles and tax records.

Usage:
    python seed_db.py
"""

import asyncio
import datetime

from backend.database import init_db, async_session_factory
from backend.models import Vehicle, TaxRecord, TaxStatus, VehicleType


SAMPLE_VEHICLES = [
    {
        "plate_number": "AB1234CD",
        "owner_name": "John Smith",
        "vehicle_type": VehicleType.CAR,
        "vehicle_brand": "Toyota",
        "vehicle_color": "White",
        "tax_status": TaxStatus.PAID,
        "tax_expiry_date": datetime.datetime(2027, 6, 30),
    },
    {
        "plate_number": "EF5678GH",
        "owner_name": "Jane Doe",
        "vehicle_type": VehicleType.CAR,
        "vehicle_brand": "Honda",
        "vehicle_color": "Black",
        "tax_status": TaxStatus.EXPIRED,
        "tax_expiry_date": datetime.datetime(2025, 12, 31),
    },
    {
        "plate_number": "IJ9012KL",
        "owner_name": "Robert Johnson",
        "vehicle_type": VehicleType.TRUCK,
        "vehicle_brand": "Ford",
        "vehicle_color": "Blue",
        "tax_status": TaxStatus.UNPAID,
    },
    {
        "plate_number": "MN3456OP",
        "owner_name": "Sarah Williams",
        "vehicle_type": VehicleType.MOTORCYCLE,
        "vehicle_brand": "Yamaha",
        "vehicle_color": "Red",
        "tax_status": TaxStatus.PAID,
        "tax_expiry_date": datetime.datetime(2027, 3, 15),
    },
    {
        "plate_number": "QR7890ST",
        "owner_name": "Michael Brown",
        "vehicle_type": VehicleType.VAN,
        "vehicle_brand": "Mercedes-Benz",
        "vehicle_color": "Silver",
        "tax_status": TaxStatus.PENDING,
    },
    {
        "plate_number": "UV2345WX",
        "owner_name": "Emily Davis",
        "vehicle_type": VehicleType.CAR,
        "vehicle_brand": "BMW",
        "vehicle_color": "Gray",
        "tax_status": TaxStatus.PAID,
        "tax_expiry_date": datetime.datetime(2027, 1, 20),
    },
    {
        "plate_number": "YZ6789AB",
        "owner_name": "David Wilson",
        "vehicle_type": VehicleType.BUS,
        "vehicle_brand": "Volvo",
        "vehicle_color": "Yellow",
        "tax_status": TaxStatus.EXPIRED,
        "tax_expiry_date": datetime.datetime(2025, 9, 1),
    },
    {
        "plate_number": "CD0123EF",
        "owner_name": "Lisa Anderson",
        "vehicle_type": VehicleType.CAR,
        "vehicle_brand": "Hyundai",
        "vehicle_color": "Green",
        "tax_status": TaxStatus.PAID,
        "tax_expiry_date": datetime.datetime(2026, 11, 30),
    },
]

SAMPLE_TAX_RECORDS = [
    # Vehicle 1 (AB1234CD) - has payment history
    {
        "vehicle_plate": "AB1234CD",
        "amount": 150.00,
        "payment_date": datetime.datetime(2025, 6, 15),
        "expiry_date": datetime.datetime(2026, 6, 30),
        "payment_method": "bank_transfer",
        "receipt_number": "TX-2025-001",
    },
    {
        "vehicle_plate": "AB1234CD",
        "amount": 155.00,
        "payment_date": datetime.datetime(2026, 6, 10),
        "expiry_date": datetime.datetime(2027, 6, 30),
        "payment_method": "card",
        "receipt_number": "TX-2026-001",
    },
    # Vehicle 4 (MN3456OP)
    {
        "vehicle_plate": "MN3456OP",
        "amount": 85.00,
        "payment_date": datetime.datetime(2026, 3, 1),
        "expiry_date": datetime.datetime(2027, 3, 15),
        "payment_method": "cash",
        "receipt_number": "TX-2026-042",
    },
    # Vehicle 6 (UV2345WX)
    {
        "vehicle_plate": "UV2345WX",
        "amount": 200.00,
        "payment_date": datetime.datetime(2026, 1, 5),
        "expiry_date": datetime.datetime(2027, 1, 20),
        "payment_method": "bank_transfer",
        "receipt_number": "TX-2026-018",
    },
]


async def seed():
    """Populate the database with sample data."""
    from sqlalchemy import select

    await init_db()

    async with async_session_factory() as session:
        # Check if data already exists
        result = await session.execute(select(Vehicle))
        existing = result.scalars().first()
        if existing:
            print("Database already has data. Skipping seed.")
            return

        # Insert vehicles
        vehicle_map = {}
        for data in SAMPLE_VEHICLES:
            vehicle = Vehicle(**data)
            session.add(vehicle)
            await session.flush()
            vehicle_map[data["plate_number"]] = vehicle.id
            print(f"  Added vehicle: {data['plate_number']} ({data['owner_name']})")

        # Insert tax records
        for record_data in SAMPLE_TAX_RECORDS:
            plate = record_data.pop("vehicle_plate")
            record = TaxRecord(
                vehicle_id=vehicle_map[plate],
                **record_data,
            )
            session.add(record)
            print(f"  Added tax record for {plate}: ${record_data['amount']}")

        await session.commit()
        print(f"\nSeeded {len(SAMPLE_VEHICLES)} vehicles and {len(SAMPLE_TAX_RECORDS)} tax records.")


if __name__ == "__main__":
    print("Seeding TaxTrack database...\n")
    asyncio.run(seed())
    print("Done!")
