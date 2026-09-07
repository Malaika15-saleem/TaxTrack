"""
Generate a synthetic dummy traffic video for testing the camera feed.

Creates a looping video with:
  - Moving road with lane markings
  - Simulated vehicles (colored rectangles) moving across the frame
  - Simulated license plate regions with text
"""

import cv2
import numpy as np
from pathlib import Path

# Plate texts to embed in the video
PLATES = [
    ("AB1234CD", (40, 80, 200)),     # White Toyota
    ("EF5678GH", (20, 20, 20)),      # Black Honda
    ("IJ9012KL", (180, 80, 30)),     # Blue Ford truck
    ("MN3456OP", (30, 30, 180)),     # Red Yamaha
    ("UV2345WX", (100, 100, 100)),   # Gray BMW
]


def generate_dummy_video(
    output_path: str,
    width: int = 640,
    height: int = 480,
    fps: int = 25,
    duration_seconds: int = 20,
):
    """
    Generate a synthetic traffic video with moving vehicles and plates.

    Parameters
    ----------
    output_path : str
        Where to save the MP4 file.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        raise RuntimeError(f"Cannot create video writer for {output_path}")

    total_frames = fps * duration_seconds

    print(f"Generating dummy video: {width}x{height} @ {fps}fps, {duration_seconds}s ({total_frames} frames)")

    # Vehicle definitions: each has speed, y-position, size, and plate info
    vehicles = []
    for i, (plate, color) in enumerate(PLATES):
        vw = np.random.randint(100, 180)
        vh = np.random.randint(70, 120)
        vehicles.append({
            "x": np.random.randint(-200, width),
            "y": 120 + i * 70,
            "w": vw,
            "h": vh,
            "speed": np.random.uniform(1.5, 4.0),
            "color": color,
            "plate": plate,
        })

    road_offset = 0

    for frame_num in range(total_frames):
        # ── Background: road ──
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (60, 60, 60)  # Dark gray road

        # Lane markings (dashed white lines, moving)
        road_offset = (road_offset + 3) % 40
        for lane_y in [160, 280, 400]:
            for dx in range(-40, width + 40, 40):
                x_start = dx + road_offset
                if 0 <= x_start < width:
                    x_end = min(x_start + 20, width)
                    cv2.line(frame, (x_start, lane_y), (x_end, lane_y), (200, 200, 200), 2)

        # Road edges
        cv2.line(frame, (0, 80), (width, 80), (255, 255, 255), 3)
        cv2.line(frame, (0, height - 40), (width, height - 40), (255, 255, 255), 3)

        # Sidewalk
        cv2.rectangle(frame, (0, 0), (width, 80), (80, 120, 80), -1)
        cv2.rectangle(frame, (0, height - 40), (width, height), (80, 120, 80), -1)

        # ── Draw vehicles ──
        for v in vehicles:
            v["x"] += v["speed"]
            if v["x"] > width + 50:
                v["x"] = -v["w"] - np.random.randint(50, 200)

            x, y, w, h = int(v["x"]), v["y"], v["w"], v["h"]
            color = v["color"]

            # Vehicle body
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, -1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)

            # Windows (darker region at top)
            win_color = tuple(max(0, c - 60) for c in color)
            cv2.rectangle(frame, (x + 10, y + 5), (x + w - 10, y + int(h * 0.35)), win_color, -1)

            # Wheels
            cv2.circle(frame, (x + int(w * 0.2), y + h), 8, (30, 30, 30), -1)
            cv2.circle(frame, (x + int(w * 0.8), y + h), 8, (30, 30, 30), -1)

            # License plate (white rectangle with text at rear)
            plate_x = x + w - 60
            plate_y = y + int(h * 0.55)
            plate_w = 55
            plate_h = 18
            cv2.rectangle(frame, (plate_x, plate_y), (plate_x + plate_w, plate_y + plate_h), (255, 255, 255), -1)
            cv2.rectangle(frame, (plate_x, plate_y), (plate_x + plate_w, plate_y + plate_h), (0, 0, 0), 1)
            cv2.putText(
                frame, v["plate"], (plate_x + 2, plate_y + 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1,
            )

        # ── HUD: frame counter ──
        cv2.putText(frame, f"Frame {frame_num}/{total_frames}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)

        writer.write(frame)

    writer.release()
    print(f"Dummy video saved: {output_path}")


if __name__ == "__main__":
    output = str(Path(__file__).resolve().parent.parent / "sample_data" / "dummy_traffic.mp4")
    generate_dummy_video(output)
