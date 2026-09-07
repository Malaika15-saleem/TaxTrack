"""
Train a YOLOv8 model for vehicle + license plate detection.
Fine-tunes yolov8n on our auto-labeled traffic video dataset.

Usage:
    python scripts/train_model.py
"""
import os
import sys
import yaml

# Ensure we can import from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def train():
    from ultralytics import YOLO

    # Load pre-trained YOLOv8n as base model
    print("Loading YOLOv8n base model...")
    model = YOLO("yolov8n.pt")

    # Dataset config path
    dataset_yaml = os.path.join(os.path.dirname(__file__), "taxtrack_dataset.yaml")

    # Verify dataset exists
    with open(dataset_yaml) as f:
        cfg = yaml.safe_load(f)

    train_dir = os.path.join(cfg["path"], "images", "train")
    val_dir = os.path.join(cfg["path"], "images", "val")

    if not os.path.exists(train_dir):
        print(f"ERROR: Train directory not found: {train_dir}")
        print("Run extract_frames.py and auto_label.py first!")
        return

    train_count = len([f for f in os.listdir(train_dir) if f.endswith(".jpg")])
    val_count = len([f for f in os.listdir(val_dir) if f.endswith(".jpg")])
    print(f"Dataset: {train_count} train + {val_count} val images")

    if train_count < 10:
        print("WARNING: Very few training images. Consider extracting more frames.")

    # Training output directory
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                              "training_data", "runs")

    # Train the model
    print("\nStarting YOLOv8 training...")
    print(f"  Model: yolov8n")
    print(f"  Classes: vehicle, license_plate")
    print(f"  Epochs: 100")
    print(f"  Image size: 640")
    print(f"  Batch: 16")
    print()

    results = model.train(
        data=dataset_yaml,
        epochs=100,
        imgsz=640,
        batch=16,
        name="taxtrack_plate_detector",
        project=output_dir,
        patience=20,       # Early stopping patience
        save=True,
        save_period=25,    # Save checkpoint every 25 epochs
        pretrained=True,
        optimizer="auto",
        lr0=0.01,
        lrf=0.01,
        # Data augmentation
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
        degrees=5.0,
        translate=0.1,
        scale=0.3,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        # Device
        device="cpu",      # Use "0" for GPU if available
        workers=2,
        verbose=True,
    )

    # Save best model path
    best_model = os.path.join(output_dir, "taxtrack_plate_detector",
                              "weights", "best.pt")
    if os.path.exists(best_model):
        print(f"\n✓ Training complete!")
        print(f"  Best model: {best_model}")

        # Copy to models directory for easy access
        models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                  "models")
        os.makedirs(models_dir, exist_ok=True)
        import shutil
        dst = os.path.join(models_dir, "taxtrack_yolov8n.pt")
        shutil.copy2(best_model, dst)
        print(f"  Copied to: {dst}")
    else:
        print(f"\n✗ Best model not found at {best_model}")

    return results


if __name__ == "__main__":
    train()
