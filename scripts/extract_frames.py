"""
Extract diverse frames from all dataset videos for YOLO training.
Samples frames uniformly with random jitter to get variety.
"""
import cv2
import os
import glob
import random

random.seed(42)

DATASET_DIR = "dataset"
OUTPUT_DIR = "training_data/raw_frames"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# How many frames to extract per video
FRAMES_PER_VIDEO = 40

video_files = sorted(glob.glob(os.path.join(DATASET_DIR, "*.mp4")))
print(f"Found {len(video_files)} videos in {DATASET_DIR}")

total_extracted = 0

for vpath in video_files:
    vname = os.path.splitext(os.path.basename(vpath))[0]
    # Shorten very long names
    vname = vname[:50].strip()

    cap = cv2.VideoCapture(vpath)
    if not cap.isOpened():
        print(f"  SKIP (cannot open): {vname}")
        continue

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames < 10:
        print(f"  SKIP (too few frames): {vname}")
        cap.release()
        continue

    # Sample uniformly spaced frames with small random jitter
    n = min(FRAMES_PER_VIDEO, total_frames // 2)
    indices = []
    step = total_frames / n
    for i in range(n):
        base = int(i * step)
        jitter = random.randint(-int(step * 0.3), int(step * 0.3))
        idx = max(0, min(total_frames - 1, base + jitter))
        indices.append(idx)

    extracted = 0
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        fname = f"{vname}_f{idx:05d}.jpg"
        fpath = os.path.join(OUTPUT_DIR, fname)
        cv2.imwrite(fpath, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        extracted += 1

    cap.release()
    total_extracted += extracted
    print(f"  {vname}: {w}x{h}, {total_frames} frames -> extracted {extracted}")

print(f"\nTotal frames extracted: {total_extracted}")
print(f"Saved to: {OUTPUT_DIR}")
