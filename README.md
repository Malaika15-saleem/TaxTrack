# TaxTrack - Vehicle Tax Verification System

A complete Python-based vehicle tax verification system that uses **YOLOv8** for vehicle/license plate detection, **PaddleOCR** for plate text recognition, **FastAPI** for the backend API, and a clean web dashboard for management.

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green) ![YOLOv8](https://img.shields.io/badge/YOLOv8-8.2-orange) ![PaddleOCR](https://img.shields.io/badge/PaddleOCR-2.8-purple)

---

## Dataset
https://drive.google.com/drive/folders/1ujVvsoqEORoH8_hGT4RpEf-u_C4G6XNl?usp=drive_link
https://drive.google.com/drive/folders/1qGF0y8r67y-oJ6ySVlc-0A1E7e_eFtiv?usp=drive_link
https://drive.google.com/drive/folders/18rQAN30EkWpyiAXsW_6NJdLux5Cv7VcJ?usp=drive_link
https://drive.google.com/drive/folders/1qp6r1A34ugbAmoWvNBzfJdG5-9V-fqUe?usp=drive_link

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌────────────┐     ┌──────────┐
│  Web Dashboard│────▶│  FastAPI      │────▶│  AI Models  │────▶│  SQLite  │
│  (HTML/CSS/JS)│◀────│  Backend      │◀────│  YOLO + OCR │     │  DB      │
└─────────────┘     └──────────────┘     └────────────┘     └──────────┘
```

**Pipeline flow:**

```
Upload Image → YOLO Detection → Plate Crop → PaddleOCR → Plate Text → DB Lookup → Tax Status
```

## Project Structure

```
TaxTrack/
├── main.py                    # FastAPI app entry point
├── config.py                  # Centralized settings (from .env)
├── requirements.txt           # Python dependencies
├── seed_db.py                 # Sample data seeder
├── .env.example               # Environment template
│
├── backend/                   # Application layer
│   ├── database.py            #   Async SQLAlchemy engine & sessions
│   ├── models.py              #   ORM models (Vehicle, ScanRecord, TaxRecord)
│   ├── schemas.py             #   Pydantic request/response schemas
│   ├── crud.py                #   Database CRUD operations
│   └── routes/
│       ├── vehicles.py        #   Vehicle & tax record endpoints
│       └── detection.py       #   Image upload + detection endpoints
│
├── ai/                        # AI / ML layer
│   ├── detector.py            #   YOLOv8 vehicle & plate detection
│   └── ocr_engine.py          #   PaddleOCR plate text recognition
│
├── templates/
│   └── index.html             # Dashboard HTML template
│
├── static/
│   ├── css/style.css          # Dashboard styles
│   └── js/dashboard.js        # Dashboard JavaScript
│
├── uploads/                   # Uploaded & annotated images
├── tests/
│   └── test_api.py            # API integration tests
│
└── sample_data/               # Sample images for testing
```

## Quick Start

### Prerequisites

- **Python 3.10+** (recommended: 3.11 or 3.12)
- **pip** (comes with Python)
- Optional: **CUDA-capable GPU** for faster inference (PaddlePaddle GPU build)

### 1. Clone and Setup

```bash
cd TaxTrack
```

### 2. Create a Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** PaddlePaddle CPU version is installed by default. For GPU support, install `paddlepaddle-gpu` instead:
> ```bash
> pip install paddlepaddle-gpu
> ```

### 4. Configure Environment

```bash
copy .env.example .env
```

Edit `.env` to customize settings (all defaults work out of the box).

### 5. Seed the Database

```bash
python seed_db.py
```

This creates `taxtrack.db` (SQLite) and inserts 8 sample vehicles with tax records.

### 6. Run the Server

```bash
python main.py
```

The server starts at **http://localhost:8000**

- **Dashboard:** http://localhost:8000/
- **API Docs (Swagger):** http://localhost:8000/docs
- **API Docs (ReDoc):** http://localhost:8000/redoc
- **Health Check:** http://localhost:8000/api/health

---

## API Endpoints

### Vehicles

| Method   | Endpoint                          | Description                    |
|----------|-----------------------------------|--------------------------------|
| `POST`   | `/api/vehicles/`                  | Register a new vehicle         |
| `GET`    | `/api/vehicles/`                  | List vehicles (filter/search)  |
| `GET`    | `/api/vehicles/{id}`              | Get vehicle by ID              |
| `GET`    | `/api/vehicles/plate/{plate}`     | Get vehicle by plate number    |
| `PUT`    | `/api/vehicles/{id}`              | Update vehicle info            |
| `DELETE` | `/api/vehicles/{id}`              | Delete vehicle + records       |

### Tax Records

| Method   | Endpoint                          | Description                    |
|----------|-----------------------------------|--------------------------------|
| `POST`   | `/api/tax/pay`                    | Record a tax payment           |
| `GET`    | `/api/tax/history/{vehicle_id}`   | Get payment history            |

### Detection & OCR

| Method   | Endpoint                          | Description                          |
|----------|-----------------------------------|--------------------------------------|
| `POST`   | `/api/detect/scan`                | Upload image for detection + OCR     |
| `POST`   | `/api/detect/scan/plate-text`     | Manual plate lookup (no image)       |

### Dashboard & System

| Method   | Endpoint                          | Description                    |
|----------|-----------------------------------|--------------------------------|
| `GET`    | `/api/stats`                      | Dashboard statistics             |
| `GET`    | `/api/health`                     | Health check                   |

---

## Database Schema

### `vehicles` table
| Column              | Type      | Description                     |
|---------------------|-----------|---------------------------------|
| id                  | INTEGER   | Primary key                     |
| plate_number        | VARCHAR   | Unique plate identifier         |
| owner_name          | VARCHAR   | Vehicle owner                   |
| vehicle_type        | ENUM      | car/truck/motorcycle/bus/van    |
| vehicle_brand       | VARCHAR   | Manufacturer                    |
| vehicle_color       | VARCHAR   | Color                           |
| tax_status          | ENUM      | paid/unpaid/expired/pending     |
| tax_expiry_date     | DATETIME  | When tax expires                |
| is_active           | BOOLEAN   | Active status                   |

### `scan_records` table
| Column               | Type     | Description                      |
|----------------------|----------|----------------------------------|
| id                   | INTEGER  | Primary key                      |
| plate_number         | VARCHAR  | Detected plate text              |
| image_path           | VARCHAR  | Path to uploaded image           |
| detection_confidence | FLOAT    | YOLO detection confidence        |
| ocr_confidence       | FLOAT    | PaddleOCR confidence             |
| vehicle_id           | INTEGER  | FK to vehicles (if found)        |
| tax_status_matched   | ENUM     | Tax status at time of scan       |
| scanned_at           | DATETIME | Timestamp                        |

### `tax_records` table
| Column          | Type     | Description                      |
|-----------------|----------|----------------------------------|
| id              | INTEGER  | Primary key                      |
| vehicle_id      | INTEGER  | FK to vehicles                   |
| amount          | FLOAT    | Payment amount                   |
| payment_date    | DATETIME | When paid                        |
| expiry_date     | DATETIME | Tax validity until               |
| payment_method  | VARCHAR  | bank_transfer/cash/card          |
| receipt_number  | VARCHAR  | Receipt reference                |

---

## AI Pipeline Details

### YOLO Detection (`ai/detector.py`)
- Uses **YOLOv8** (Ultralytics) for vehicle detection
- Default model: `yolov8n.pt` (nano, fastest) — auto-downloads on first run
- Detects cars, trucks, motorcycles, buses using COCO class IDs
- **Plate detection:** Uses a heuristic to estimate plate region from vehicle bounding boxes
- **Production tip:** Fine-tune YOLOv8 on a license plate dataset (e.g., from Roboflow) for direct plate detection, then set `YOLO_MODEL_PATH` in `.env`

### PaddleOCR Recognition (`ai/ocr_engine.py`)
- Uses **PaddleOCR** for text extraction from plate crops
- **Preprocessing:** Grayscale conversion → adaptive thresholding → resize to optimal height
- **Postprocessing:** Alphanumeric filtering, uppercase normalization
- Supports angle classification for tilted plates

---

## Web Dashboard

The dashboard provides three tabs:

1. **Dashboard** — Statistics overview (total vehicles, tax status breakdown, scan count) and recent scan history
2. **Vehicles** — Full CRUD table with search, status filtering, and add/delete functionality
3. **Scan** — Image upload for AI-powered plate detection + OCR, or manual plate number lookup

---

## Testing

```bash
pytest tests/ -v
```

Tests cover:
- Health check endpoint
- Vehicle CRUD (create, list, get by plate, duplicate detection)
- Dashboard statistics
- Manual plate lookup (found and not-found cases)

---

## Configuration

All settings are in `.env` (copy from `.env.example`):

| Variable                   | Default           | Description                       |
|---------------------------|-------------------|-----------------------------------|
| `HOST`                    | `0.0.0.0`         | Server bind address               |
| `PORT`                    | `8000`            | Server port                       |
| `DATABASE_URL`            | `sqlite+aiosqlite:///./taxtrack.db` | Database connection |
| `YOLO_MODEL_PATH`         | `yolov8n.pt`      | Path to YOLO model weights        |
| `YOLO_CONFIDENCE_THRESHOLD` | `0.45`          | Minimum detection confidence      |
| `PADDLEOCR_LANG`          | `en`              | OCR language                      |
| `PADDLEOCR_USE_GPU`       | `false`           | Enable GPU for OCR                |
| `MAX_UPLOAD_SIZE_MB`      | `10`              | Max image upload size             |

---

## Production Considerations

1. **Use a fine-tuned YOLO model** trained specifically on license plates for much better detection accuracy
2. **Switch to PostgreSQL** by changing `DATABASE_URL` in `.env`
3. **Add authentication** — JWT or API key middleware
4. **Add rate limiting** for the detection endpoints
5. **Use Redis** for caching scan results and session management
6. **Deploy with Gunicorn + UvicornWorker** behind Nginx
7. **Add image storage** to S3/MinIO instead of local filesystem
8. **Enable HTTPS** via reverse proxy or cloud provider

---

## License

MIT License
