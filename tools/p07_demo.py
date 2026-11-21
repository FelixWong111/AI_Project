"""Create visibly synthetic offline UI fixtures, never a dataset or model result."""
from pathlib import Path
import cv2
import numpy as np


def create(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    image = np.full((480, 640, 3), 230, np.uint8)
    cv2.putText(image, "SYNTHETIC UI FIXTURE - NO REAL PARTS", (20, 240), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 180), 2)
    cv2.imencode('.png', image)[1].tofile(directory / "mock-input.png")
    writer = cv2.VideoWriter(str(directory / "mock-input.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 10, (640, 480))
    if not writer.isOpened():
        raise OSError("MJPG unavailable")
    try:
        for n in range(30):
            frame = image.copy()
            cv2.putText(frame, str(n), (20, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 180), 2)
            writer.write(frame)
    finally:
        writer.release()


if __name__ == "__main__":
    create(Path("tmp/p07-demo"))
    print("Created tmp/p07-demo/mock-input.png and mock-input.avi (synthetic)")
