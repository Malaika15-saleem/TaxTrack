"""
Auto-label extracted frames for YOLO training.
Uses pre-trained YOLOv8 to detect vehicles, then heuristics + Tesseract
to find license plate regions. Outputs YOLO-format annotations.

Classes:
  0 = vehicle
  1 = license_plate
"""
import cv2
import os
import glob
import numpy as np
import re

# Will be imported after ultralytics is installed
try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False
    print("WARNING: ultralytics not installed yet. Install with: pip install ultralytics")

try:
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    HAS_TESSERACT = True
except Exception:
    HAS_TESSERACT = False
    print("WARNING: Tesseract not available")


RAW_DIR = "training_data/raw_frames"
DATASET_DIR = "training_data/yolo_dataset"

# YOLO dataset structure
for split in ["images/train", "images/val", "labels/train", "labels/val"]:
    os.makedirs(os.path.join(DATASET_DIR, split), exist_ok=True)

VAL_RATIO = 0.2  # 20% validation

# COCO vehicle class IDs
VEHICLE_IDS = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def find_plate_in_crop(crop_img):
    """
    Try to find a license plate region within a vehicle crop.
    Returns (x, y, w, h) relative to crop, or None.
    """
    if crop_img is None or crop_img.size == 0:
        return None

    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Resize for processing
    if w < 100:
        scale = 200 / w
        gray = cv2.resize(gray, (200, int(h * scale)))
        h, w = gray.shape

    # Edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Dilate to connect edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_plate = None
    best_score = 0

    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)

        # Plate aspect ratio: typically 2:1 to 5:1
        aspect = cw / max(ch, 1)
        if aspect < 1.5 or aspect > 6.0:
            continue

        # Size constraints (plate should be reasonable fraction of vehicle)
        area = cw * ch
        frame_area = w * h
        if area < frame_area * 0.01 or area > frame_area * 0.3:
            continue

        # Score based on aspect ratio closeness to typical plate (3:1)
        score = 1.0 - abs(aspect - 3.5) / 3.5

        # Check for text-like content using Tesseract
        if HAS_TESSERACT:
            plate_roi = gray[y:y+ch, x:x+cw]
            if plate_roi.size > 0:
                try:
                    config = "--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                    text = pytesseract.image_to_string(plate_roi, config=config).strip()
                    text = re.sub(r"[^A-Z0-9]", "", text.upper())
                    if len(text) >= 4:
                        score += 2.0  # Big bonus for readable text
                except Exception:
                    pass

        if score > best_score:
            best_score = score
            best_plate = (x, y, cw, ch)

    return best_plate if best_score > 0.3 else None


def auto_label():
    if not HAS_YOLO:
        print("ERROR: ultralytics required. Install first.")
        return

    frames = sorted(glob.glob(os.path.join(RAW_DIR, "*.jpg")))
    print(f"Found {len(frames)} frames to label")

    # Load pre-trained YOLOv8n for auto-detection
    print("Loading YOLOv8n for vehicle detection...")
    model = YOLO("yolov8n.pt")

    labeled = 0
    skipped = 0

    for i, fpath in enumerate(frames):
        fname = os.path.basename(fpath)
        img = cv2.imread(fpath)
        if img is None:
            continue

        img_h, img_w = img.shape[:2]

        # Determine split
        is_val = (i % int(1 / VAL_RATIO) == 0)
        split = "val" if is_val else "train"

        # Run YOLO detection
        results = model(img, conf=0.35, verbose=False)

        annotations = []  # (class_id, cx, cy, w, h) in YOLO format (normalized)

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for j in range(len(boxes)):
                cls_id = int(boxes.cls[j].item())
                conf = float(boxes.conf[j].item())
                x1, y1, x2, y2 = boxes.xyxy[j].tolist()

                # Vehicle classes
                if cls_id in VEHICLE_IDS:
                    # YOLO format: class cx cy w h (all normalized 0-1)
                    cx = ((x1 + x2) / 2) / img_w
                    cy = ((y1 + y2) / 2) / img_h
                    bw = (x2 - x1) / img_w
                    bh = (y2 - y1) / img_h
                    annotations.append((0, cx, cy, bw, bh))

                    # Try to find plate in vehicle crop
                    vx1, vy1, vx2, vy2 = int(x1), int(y1), int(x2), int(y2)
                    vx1, vy1 = max(0, vx1), max(0, vy1)
                    vx2, vy2 = min(img_w, vx2), min(img_h, vy2)
                    crop = img[vy1:vy2, vx1:vx2]

                    plate = find_plate_in_crop(crop)
                    if plate is not None:
                        px, py, pw, ph = plate
                        # Convert plate coords to full image coords
                        abs_px = vx1 + px
                        abs_py = vy1 + py

                        pcx = (abs_px + pw / 2) / img_w
                        pcy = (abs_py + ph / 2) / img_h
                        pbw = pw / img_w
                        pbh = ph / img_h

                        # Clamp to [0, 1]
                        pcx = max(0, min(1, pcx))
                        pcy = max(0, min(1, pcy))
                        pbw = max(0, min(1, pbw))
                        pbh = max(0, min(1, pbh))

                        annotations.append((1, pcx, pcy, pbw, pbh))

        if not annotations:
            skipped += 1
            continue

        # Copy image to dataset
        img_dst = os.path.join(DATASET_DIR, "images", split, fname)
        cv2.imwrite(img_dst, img, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Write YOLO label file
        label_name = os.path.splitext(fname)[0] + ".txt"
        label_path = os.path.join(DATASET_DIR, "labels", split, label_name)
        with open(label_path, "w") as f:
            for cls, cx, cy, bw, bh in annotations:
                f.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

        labeled += 1
        vehicles = sum(1 for a in annotations if a[0] == 0)
        plates = sum(1 for a in annotations if a[0] == 1)

        if (i + 1) % 20 == 0 or i == 0:
            print(f"  [{i+1}/{len(frames)}] {fname}: "
                  f"{vehicles} vehicles, {plates} plates → {split}")

    print(f"\nLabeling complete:")
    print(f"  Labeled: {labeled} frames")
    print(f"  Skipped (no detections): {skipped} frames")
    print(f"  Dataset: {DATASET_DIR}")

    # Count splits
    train_imgs = len(glob.glob(os.path.join(DATASET_DIR, "images/train/*.jpg")))
    val_imgs = len(glob.glob(os.path.join(DATASET_DIR, "images/val/*.jpg")))
    print(f"  Train: {train_imgs} images")
    print(f"  Val:   {val_imgs} images")


if __name__ == "__main__":
    auto_label()
