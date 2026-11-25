"""P07 session evidence, distinct from P08 batch/evaluation outputs."""
from __future__ import annotations

import csv
import json
import platform
import threading
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

CAPTURE_FIELDS = "session_id,frame_id,timestamp,camera_index,capture_backend,width,height,frame_read_ms,capture_fps,read_success".split(",")
LATENCY_FIELDS = "frame_id,source,mode,model,model_version,width,height,inference_ms,processing_ms,processing_fps,status,expected_counts".split(",")
IMAGE_FIELDS = "image_id,model_id,status,predicted_label,prediction_score,score_type,non_catalogue_count".split(",")
OBJECT_FIELDS = "image_id,model_id,detection_id,class_id,xmin,ymin,xmax,ymax,score".split(",")
COUNT_FIELDS = "image_id,class_id,observed_count,expected_count,difference".split(",")


def dump(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


class Journal:
    def __init__(self, root: Path, config: dict):
        self.session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid4().hex[:8]
        self.path = root / self.session_id
        self.path.mkdir(parents=True)
        self.lock = threading.Lock()
        self.next_frame = 0
        self.rows = []
        self.capture_file = (self.path / "capture.csv").open("w", newline="", encoding="utf-8")
        self.capture_writer = csv.writer(self.capture_file)
        self.capture_writer.writerow(CAPTURE_FIELDS)
        self.latency_file = (self.path / "latency.csv").open("w", newline="", encoding="utf-8")
        self.latency_writer = csv.DictWriter(self.latency_file, fieldnames=LATENCY_FIELDS)
        self.latency_writer.writeheader()
        self.hardware = dict(platform=platform.platform(), processor=platform.processor(), python=platform.python_version(),
                             opencv=cv2.__version__, numpy=np.__version__)
        dump(self.path / "session.json", dict(session_id=self.session_id, config=config, hardware=self.hardware,
             timing_boundary="frame returned by read -> imshow + waitKey returned; OS/compositor/photon latency unmeasured",
             frame_id="P07 read-attempt counter; not phone frame ID, image_id, or scene_id"))

    def capture(self, kind, value, backend, image, read_ms, fps, ok):
        with self.lock:
            self.next_frame += 1
            h, w = image.shape[:2] if image is not None else (0, 0)
            self.capture_writer.writerow([self.session_id, self.next_frame, datetime.now(timezone.utc).isoformat(),
                                          value if kind == "camera" else "", backend, w, h, read_ms, fps, int(ok)])
            self.capture_file.flush()
            return self.next_frame

    def latency(self, row):
        self.rows.append(row)
        self.latency_writer.writerow(row)
        self.latency_file.flush()

    def summarize(self):
        groups = {}
        for row in self.rows:
            key = (row["mode"], row["model"], row["model_version"], row["source"], row["width"], row["height"])
            groups.setdefault(key, {})[row["frame_id"]] = row
        summaries = []
        for key, unique in groups.items():
            rows = list(unique.values())
            good = [r for r in rows if r["status"] in ("PASS", "FAIL", "REVIEW")]
            values = [r["processing_ms"] for r in good]
            summaries.append(dict(group=key, frames=len(values), failed_or_unsupported=len(rows)-len(good),
                                  median_ms=float(np.median(values)) if values else None,
                                  p95_ms=float(np.percentile(values, 95)) if values else None,
                                  evidence="MOCK_DEBUG_ONLY" if key[0] == "mock" else
                                  ("LIVE_PROXY_20_FRAMES" if key[0] == "real" and key[3].startswith("camera:") and len(values) >= 20 else "INSUFFICIENT_LIVE_EVIDENCE")))
        dump(self.path / "latency_summary.json", summaries)

    def close(self):
        self.summarize()
        self.capture_file.close()
        self.latency_file.close()


def write_diagnostics(directory: Path, images: list, metadata: dict):
    """Additive evidence sidecar; keep the course output contracts unchanged."""
    rows = [dict(image_id=iid, expected_counts=dict(expected),
                 result={k: v for k, v in result.items() if not k.startswith('_')})
            for iid, expected, result in images]
    dump(directory / 'diagnostics.json', dict(schema_version=1, model=metadata, images=rows))


def export_result(directory: Path, image_id: str, model: str, result: dict, expected: dict, run_config: dict):
    directory.mkdir(parents=True, exist_ok=False)
    def write(name, fields, rows):
        with (directory / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    write("images.csv", IMAGE_FIELDS, [dict(image_id=image_id, model_id=model, **{k: result.get(k) for k in IMAGE_FIELDS[2:]})])
    if result.get("counts") is not None:
        fields = ('detection_id','class_id','xmin','ymin','xmax','ymax','score')
        write("objects.csv", OBJECT_FIELDS, [dict(image_id=image_id, model_id=model,
              **{k:obj[k] for k in fields}) for obj in result["objects"]])
        write("counts.csv", COUNT_FIELDS, [dict(image_id=image_id, class_id=c, observed_count=result["counts"][str(c)],
              expected_count=0 if c == 0 else expected[str(c)],
              difference=result["counts"][str(c)]-(0 if c == 0 else expected[str(c)])) for c in range(6)])
    dump(directory / "run_config.json", dict(run_config, per_image_expected_counts={image_id: expected}))
    if model == 'baseline3':
        write_diagnostics(directory, [(image_id, expected, result)], run_config)


class Recorder:
    def __init__(self, path: Path, fps: float, size: tuple):
        self.path, self.fps, self.size, self.frames = path, fps, size, 0
        self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        if not self.writer.isOpened():
            self.writer.release()
            raise OSError("MP4 encoder unavailable")

    def write(self, image):
        if (image.shape[1], image.shape[0]) != self.size:
            raise ValueError("Recording resolution changed; stop recording first")
        self.writer.write(image)
        self.frames += 1

    def close(self):
        self.writer.release()
        dump(self.path.with_suffix(".json"), dict(codec="mp4v", fps=self.fps, size=self.size, frames=self.frames,
             timing="constant-rate encoding of submitted GUI frames; not original acquisition timestamps"))
