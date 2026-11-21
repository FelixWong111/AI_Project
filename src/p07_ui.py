"""Result presenter and one background inference worker."""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from .p07_bridge import CLASSES, validate_result


class InferenceWorker:
    def __init__(self):
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="p07-inference")
        self.future = None

    @property
    def busy(self):
        return self.future is not None

    def submit(self, backend, model, packet, expected, scenario, generation, previous=None):
        if self.busy:
            return False
        def run():
            start = time.perf_counter()
            try:
                if previous is None:
                    backend.synchronize()
                    start = time.perf_counter()
                    result = backend.infer(packet.image, expected, scenario) if backend.is_mock else backend.infer(packet.image, expected)
                    backend.synchronize()
                    inference_ms = (time.perf_counter() - start) * 1000
                else:
                    result = backend.rejudge(previous, expected)
                    inference_ms = None
                result = validate_result(result, model, packet.image.shape)
            except Exception as exc:
                result = dict(status="ERROR", predicted_label=None, prediction_score=None, score_type=None,
                              non_catalogue_count=None, counts=None, objects=[], message=str(exc))
                inference_ms = None
            return dict(result=result, packet=packet, expected=dict(expected), model=model, generation=generation,
                        inference_ms=inference_ms, backend=backend, rejudge=previous is not None)
        self.future = self.pool.submit(run)
        return True

    def poll(self):
        if self.future is not None and self.future.done():
            future, self.future = self.future, None
            return future.result()
        return None

    def close(self):
        self.pool.shutdown(wait=True)


def present(packet, result, expected, model, mode, info):
    """Render boxes on the exact inferred frame, never on a newer preview."""
    if packet is None:
        image = np.zeros((480, 640, 3), np.uint8)
    else:
        image = packet.image.copy()
    for obj in (result or {}).get("objects", []):
        x1, y1, x2, y2 = (int(obj[k]) for k in ("xmin", "ymin", "xmax", "ymax"))
        color = (40, 40, 255) if obj["class_id"] == 0 else (80, 220, 80)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f'{CLASSES[obj["class_id"]]} {obj["score"]:.3f}', (x1, max(18, y1+18)), cv2.FONT_HERSHEY_SIMPLEX, .48, color, 1)
    scale = min(960 / image.shape[1], 720 / image.shape[0], 1.0)
    image = cv2.resize(image, (max(1, int(image.shape[1]*scale)), max(1, int(image.shape[0]*scale))))
    panel = np.zeros((max(720, image.shape[0]), 500, 3), np.uint8)
    status = (result or {}).get("status", "PREVIEW")
    color = (80, 240, 80) if status == "PASS" else (50, 80, 255)
    lines = [f'{"MOCK INFERENCE - TEST RESULTS" if mode == "mock" else "REAL BACKEND"}',
             f'{model} | {status}', f'Label: {(result or {}).get("predicted_label")}',
             f'Score: {(result or {}).get("prediction_score")} ({(result or {}).get("score_type") or "N/A"})',
             'MOCK scores are synthetic' if mode == "mock" else 'Score meaning: see bridge metadata',
             'Class              Obs / Exp / Diff']
    counts = (result or {}).get("counts")
    for c in range(1, 6):
        obs = counts[str(c)] if counts is not None else None
        lines.append(f'{CLASSES[c]:17} {obs if obs is not None else "N/A"} / {expected[str(c)]} / {obs-expected[str(c)] if obs is not None else "N/A"}')
    lines += [f'ID 0: {(result or {}).get("non_catalogue_count", "N/A")}',
              'WARNING: NON-CATALOGUE' if counts and counts['0'] else '', *info,
              str((result or {}).get("message", ""))[:58],
              'SPACE detect/pause | 1..4 model | n mock case',
              'c camera | x close | +/- camera index',
              'i image | v video | e expected JSON',
              's snapshot + CSV | r record | q quit']
    for i, line in enumerate(lines):
        cv2.putText(panel, str(line)[:66], (10, 25 + i*24), cv2.FONT_HERSHEY_SIMPLEX, .48, color if i < 2 else (225, 225, 225), 1)
    canvas = np.zeros((panel.shape[0], image.shape[1]+panel.shape[1], 3), np.uint8)
    canvas[:image.shape[0], :image.shape[1]] = image
    canvas[:, image.shape[1]:] = panel
    # Visible watermark also covers the source view and all saved recordings.
    if mode == "mock":
        cv2.putText(canvas, 'MOCK', (12, canvas.shape[0]-24), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3)
    return canvas
