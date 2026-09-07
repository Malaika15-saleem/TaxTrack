"""
Basic tests for the TaxTrack API.
Run with:  pytest tests/ -v
"""

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from main import app
from backend.database import init_db, engine, Base


@pytest_asyncio.fixture(scope="session")
async def setup_db():
    """Create tables before tests and drop after."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def client(setup_db):
    """Create an async test client."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ──────────────────────────────────────────────
# Health Check
# ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"


# ──────────────────────────────────────────────
# Vehicle CRUD
# ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_create_vehicle(client):
    payload = {
        "plate_number": "TEST001",
        "owner_name": "Test User",
        "vehicle_type": "car",
        "vehicle_brand": "Tesla",
        "vehicle_color": "Red",
        "tax_status": "paid",
    }
    resp = await client.post("/api/vehicles/", json=payload)
    assert resp.status_code == 201
    data = resp.json()
    assert data["plate_number"] == "TEST001"
    assert data["owner_name"] == "Test User"


@pytest.mark.asyncio
async def test_create_duplicate_vehicle(client):
    payload = {
        "plate_number": "TEST001",
        "owner_name": "Another User",
        "vehicle_type": "car",
    }
    resp = await client.post("/api/vehicles/", json=payload)
    assert resp.status_code == 409


@pytest.mark.asyncio
async def test_list_vehicles(client):
    resp = await client.get("/api/vehicles/")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) >= 1


@pytest.mark.asyncio
async def test_get_vehicle_by_plate(client):
    resp = await client.get("/api/vehicles/plate/TEST001")
    assert resp.status_code == 200
    assert resp.json()["plate_number"] == "TEST001"


@pytest.mark.asyncio
async def test_get_nonexistent_vehicle(client):
    resp = await client.get("/api/vehicles/plate/ZZZZZZZ")
    assert resp.status_code == 404


# ──────────────────────────────────────────────
# Stats
# ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_dashboard_stats(client):
    resp = await client.get("/api/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_vehicles" in data
    assert "total_scans" in data


# ──────────────────────────────────────────────
# Manual Plate Lookup
# ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_plate_lookup(client):
    resp = await client.post("/api/detect/scan/plate-text?plate=TEST001")
    assert resp.status_code == 200
    data = resp.json()
    assert data["plate_number"] == "TEST001"
    assert data["vehicle_found"] is True


@pytest.mark.asyncio
async def test_plate_lookup_unknown(client):
    resp = await client.post("/api/detect/scan/plate-text?plate=UNKNOWN999")
    assert resp.status_code == 200
    data = resp.json()
    assert data["vehicle_found"] is False
