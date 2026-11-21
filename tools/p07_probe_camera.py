"""Explicit OpenCV device-index probe; never identifies the physical phone model."""
import argparse
import json
import time
from pathlib import Path

import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--indices", type=int, nargs="+", required=True)
parser.add_argument("--output", type=Path, default=Path("docs/p07/camera-probe.json"))
args = parser.parse_args()
records = []
for index in args.indices:
    for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF):
        cap = cv2.VideoCapture(index, backend)
        row = dict(index=index, backend=backend, opened=cap.isOpened(), read_success=False)
        try:
            if cap.isOpened():
                start = time.perf_counter()
                ok, frame = cap.read()
                row.update(read_success=ok, frame_read_ms=(time.perf_counter()-start)*1000,
                           resolution=list(frame.shape[:2][::-1]) if ok else None,
                           device_identity="TBD; verify Camo preview manually")
        finally:
            cap.release()
        records.append(row)
        print(row, flush=True)
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(records, indent=2), encoding="utf-8")
