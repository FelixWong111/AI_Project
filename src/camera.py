"""P07 inputs. Device handles are used and released by the capture thread only."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


class Rate:
    def __init__(self):
        self.times = []

    def tick(self, now: float) -> float:
        self.times.append(now)
        self.times = self.times[-60:]
        return self.fps

    @property
    def fps(self) -> float:
        return (len(self.times) - 1) / (self.times[-1] - self.times[0]) if len(self.times) > 1 and self.times[-1] > self.times[0] else 0.0


@dataclass
class Frame:
    frame_id: int
    image: np.ndarray
    available_at: float
    capture_fps: float
    source: str


class FrameProvider:
    def __init__(self, kind: str, value, config: dict, journal, capture_factory=None):
        self.kind, self.value, self.config, self.journal = kind, value, config, journal
        self.factory = capture_factory or cv2.VideoCapture
        self.stop_event = threading.Event()
        self.latest: Optional[Frame] = None
        self.error = "Opening source"
        self.finished = False
        self.rate = Rate()
        self.cap = None
        self.backend = "image"
        self.thread = threading.Thread(target=self._run, name="p07-capture", daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _open(self):
        if self.kind == "video":
            self.cap = self.factory(str(self.value))
        else:
            choices = {"dshow": [cv2.CAP_DSHOW], "msmf": [cv2.CAP_MSMF], "any": [cv2.CAP_ANY],
                       "auto": [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]}[self.config["camera_backend"]]
            for backend in choices:
                self.cap = self.factory(int(self.value), backend)
                if self.cap.isOpened():
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config["width"])
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config["height"])
                    break
                self.cap.release()
        if not self.cap.isOpened():
            raise OSError("Cannot open " + self.kind + ": " + str(self.value))
        try:
            self.backend = self.cap.getBackendName()
        except (cv2.error, AttributeError):
            self.backend = self.config["camera_backend"]

    def _release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def _run(self):
        try:
            if self.kind == "image":
                start = time.perf_counter()
                image = cv2.imdecode(np.fromfile(Path(self.value), dtype=np.uint8), cv2.IMREAD_COLOR)
                self._publish(image is not None, image, start)
                if image is None:
                    self.error = "Unreadable image: " + str(self.value)
                self.finished = True
                return
            while not self.stop_event.is_set():
                start = time.perf_counter()
                try:
                    if self.cap is None:
                        self._open()
                    # frame_read_ms excludes open/reconnect negotiation.
                    start = time.perf_counter()
                    ok, image = self.cap.read()
                    self._publish(ok, image, start)
                    if not ok:
                        self._release()
                        if self.kind == "video":
                            self.error = "Video EOF or decode failure"
                            break
                        self.error = "Camera disconnected / read failed; reconnecting"
                        self.latest = None
                        self.rate = Rate()
                        self.stop_event.wait(self.config["reconnect_seconds"])
                    elif self.kind == "video":
                        # Playback pacing is separate from read-call duration.
                        fps = self.cap.get(cv2.CAP_PROP_FPS)
                        self.stop_event.wait(1.0 / fps if fps and fps > 0 else 1.0 / 30)
                except (OSError, cv2.error) as exc:
                    self.error = str(exc)
                    self._publish(False, None, time.perf_counter())
                    self.latest = None
                    self._release()
                    if self.kind == "video":
                        break
                    self.stop_event.wait(self.config["reconnect_seconds"])
        except (OSError, ValueError) as exc:
            self.error = str(exc)
            self._publish(False, None, time.perf_counter())
        finally:
            self._release()
            self.finished = True

    def _publish(self, ok: bool, image, start: float):
        now = time.perf_counter()
        frame_id = self.journal.capture(self.kind, self.value, self.backend, image if ok else None,
                                        (now - start) * 1000, self.rate.tick(now) if ok else 0, ok)
        if ok:
            self.latest = Frame(frame_id, image, now, self.rate.fps, self.kind + ":" + str(self.value))
            self.error = ""

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=3)
        if self.thread.is_alive():
            raise RuntimeError("Capture driver blocked in read/open; close device in Camo and retry shutdown")
