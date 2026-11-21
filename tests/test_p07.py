from __future__ import annotations

import csv
import json
import tempfile
import time
import unittest
from pathlib import Path

import cv2
import numpy as np

from app import load_expected
from src.camera import Frame, FrameProvider
from src.p07_devices import select_camera
from src.p07_bridge import MODELS, MockBackend, load_backend, validate_counts, validate_result
from src.p07_output import CAPTURE_FIELDS, COUNT_FIELDS, IMAGE_FIELDS, OBJECT_FIELDS, Journal, Recorder, export_result
from src.p07_ui import InferenceWorker, present
from tools.p07_demo import create

CONFIG = json.loads(Path("config/p07.json").read_text())
ONES = {c: 1 for c in "12345"}
ZERO = dict.fromkeys("12345", 0)
IMAGE = np.zeros((480, 640, 3), np.uint8)


class MockIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.backend = MockBackend("baseline3", CONFIG)

    def infer(self, case, counts=ONES):
        result = self.backend.infer(IMAGE, counts, case)
        return validate_result(result, "baseline3", IMAGE.shape)

    def test_U01(self):
        self.assertEqual(self.infer("pass")["status"], "PASS")

    def test_U02(self):
        for case in ("missing", "extra"):
            self.assertEqual(self.infer(case)["status"], "FAIL")
        self.assertEqual(self.infer("pass", ZERO)["status"], "FAIL")

    def test_U03(self):
        result = self.infer("id0")
        self.assertEqual((result["non_catalogue_count"], result["status"]), (1, "FAIL"))

    def test_U04_no_reinfer_or_observation_change(self):
        for case in ("pass", "id0"):
            raw = self.infer(case)
            new = self.backend.rejudge(raw, ZERO)
            self.assertIs(new["objects"], raw["objects"])
            self.assertEqual(new["counts"], raw["counts"])
            self.assertEqual(new["status"], "FAIL")
            self.assertEqual(self.backend.rejudge(new, ONES)["status"], "PASS" if case == "pass" else "FAIL")

    def test_U05(self):
        self.assertEqual(self.infer("empty", ZERO)["status"], "PASS")
        result = self.infer("only_id0", ZERO)
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(len(result["objects"]), 1)

    def test_U06(self):
        self.assertEqual(self.infer("review")["status"], "REVIEW")
        self.assertEqual(self.backend.rejudge(self.infer("review"), ZERO)["status"], "REVIEW")
        for model in MODELS[:2]:
            backend = MockBackend(model, CONFIG)
            raw = backend.infer(IMAGE, ZERO)
            self.assertEqual(raw["status"], "UNSUPPORTED_COUNTS")
            self.assertIsNone(raw["prediction_score"])
            self.assertIsNone(raw["predicted_label"])
            self.assertIsNone(raw["counts"])
            self.assertEqual(backend.rejudge(raw, ONES)["status"], "PASS")

    def test_invalid_counts(self):
        for invalid in ({}, {**ONES, "0": 0}, {**ONES, "1": True}, {**ONES, "1": -1}, {**ONES, "1": 1.5}):
            with self.assertRaises(ValueError):
                validate_counts(invalid)

    def test_invalid_frame(self):
        for invalid in (None, np.zeros((10, 10), np.uint8), IMAGE.astype(float)):
            with self.assertRaises(ValueError):
                self.backend.infer(invalid, ONES)

    def test_real_never_falls_back(self):
        with self.assertRaisesRegex(RuntimeError, "no MOCK fallback"):
            load_backend("real", "final", CONFIG)

    def test_camera_selection_matches_name_and_backend(self):
        devices = [dict(index=0, name="Camo", backend="dshow"),
                   dict(index=1, name="HP Camera", backend="dshow"),
                   dict(index=3, name="Camo", backend="msmf")]
        self.assertEqual(select_camera("Camo", devices=devices)["index"], 0)
        self.assertEqual(select_camera("camo", "msmf", devices)["index"], 3)
        with self.assertRaises(ValueError): select_camera("absent", devices=devices)
        with self.assertRaises(ValueError): select_camera("Camera", devices=[
            dict(index=0, name="Camera A", backend="dshow"), dict(index=1, name="Camera B", backend="dshow")])

    def test_image_empty_fixed_counts_is_fail(self):
        result = MockBackend("baseline1", CONFIG).infer(IMAGE, ONES, "empty")
        self.assertEqual(result["status"], "FAIL")

    def test_review_without_reliable_counts(self):
        raw = dict(status="REVIEW", predicted_label=None, counts=None, objects=[])
        self.assertEqual(validate_result(raw, "final", IMAGE.shape)["status"], "REVIEW")

    def test_invalid_result(self):
        result = self.infer("pass")
        result["objects"][0]["xmax"] = 10000
        with self.assertRaises(ValueError):
            validate_result(result, "baseline3", IMAGE.shape)

    def test_ui_for_all_models(self):
        for model in MODELS:
            result = MockBackend(model, CONFIG).infer(IMAGE, ONES)
            canvas = present(Frame(1, IMAGE, 0, 0, "image"), result, ONES, model, "mock", [])
            self.assertEqual(canvas.shape[1], 1140)
            self.assertGreater(np.count_nonzero(canvas), 100)

    def test_worker_error_and_rejudge(self):
        worker = InferenceWorker()
        packet = Frame(1, IMAGE, time.perf_counter(), 0, "image")
        try:
            worker.submit(self.backend, "baseline3", packet, ONES, "error", 2)
            deadline = time.perf_counter()+3
            result = None
            while result is None and time.perf_counter() < deadline:
                result = worker.poll()
                time.sleep(.001)
            self.assertEqual(result["result"]["status"], "ERROR")
            worker.submit(self.backend, "baseline3", packet, ZERO, "pass", 2, self.infer("pass"))
            while (result := worker.poll()) is None:
                time.sleep(.001)
            self.assertTrue(result["rejudge"])
            self.assertIsNone(result["inference_ms"])
        finally:
            worker.close()


class InputOutputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)
        self.journal = Journal(self.path, CONFIG)

    def tearDown(self):
        self.journal.close()
        self.temp.cleanup()

    def wait_provider(self, provider):
        deadline = time.perf_counter()+5
        while not provider.finished and time.perf_counter() < deadline:
            time.sleep(.01)
        provider.close()

    def test_image_video_and_csv(self):
        create(self.path)
        for kind, name in (("image", "mock-input.png"), ("video", "mock-input.avi")):
            provider = FrameProvider(kind, self.path/name, CONFIG, self.journal).start()
            self.wait_provider(provider)
            self.assertIsNotNone(provider.latest)
            self.assertEqual(provider.latest.image.shape, IMAGE.shape)
        with (self.journal.path/"capture.csv").open(encoding="utf-8") as stream:
            rows = list(csv.reader(stream))
        self.assertEqual(rows[0], CAPTURE_FIELDS)
        self.assertGreater(len(rows), 30)
        self.assertEqual([int(r[1]) for r in rows[1:]], list(range(1, len(rows))))

    def test_bad_image(self):
        provider = FrameProvider("image", self.path/"absent.png", CONFIG, self.journal).start()
        self.wait_provider(provider)
        self.assertIsNone(provider.latest)
        self.assertTrue(provider.error)

    def test_fake_camera_disconnect_reconnect_and_release(self):
        handles = []
        class Capture:
            def __init__(self, *args):
                self.reads = 0
                self.released = False
                handles.append(self)
            def isOpened(self): return True
            def set(self, *args): pass
            def getBackendName(self): return "FAKE"
            def read(self):
                self.reads += 1
                time.sleep(.002)
                return (False, None) if len(handles) == 1 else (True, IMAGE)
            def release(self): self.released = True
        config = dict(CONFIG, reconnect_seconds=.01)
        provider = FrameProvider("camera", 7, config, self.journal, Capture).start()
        deadline = time.perf_counter()+2
        while provider.latest is None and time.perf_counter() < deadline:
            time.sleep(.01)
        provider.close()
        self.assertIsNotNone(provider.latest)
        self.assertGreaterEqual(len(handles), 2)
        self.assertTrue(all(h.released for h in handles))

    def test_camera_open_failure(self):
        class Closed:
            def __init__(self, *args): self.released = False
            def isOpened(self): return False
            def release(self): self.released = True
        provider = FrameProvider("camera", 3, CONFIG, self.journal, Closed).start()
        time.sleep(.05)
        provider.close()
        self.assertIn("Cannot open camera", provider.error)

    def test_exports(self):
        for model in MODELS:
            result = MockBackend(model, CONFIG).infer(IMAGE, ONES)
            directory = self.path/model
            export_result(directory, "case.png", model, result, ONES, {"evidence_mode": "mock"})
            with (directory/"images.csv").open() as stream:
                self.assertEqual(next(csv.reader(stream)), IMAGE_FIELDS)
            if model in MODELS[:2]:
                self.assertFalse((directory/"counts.csv").exists())
            else:
                with (directory/"counts.csv").open() as stream:
                    rows = list(csv.reader(stream))
                self.assertEqual(rows[0], COUNT_FIELDS)
                self.assertEqual(len(rows), 7)
                with (directory/"objects.csv").open() as stream:
                    self.assertEqual(next(csv.reader(stream)), OBJECT_FIELDS)

    def test_recording(self):
        recorder = Recorder(self.path/"mock.mp4", 10, (640, 480))
        for _ in range(3): recorder.write(IMAGE)
        recorder.close()
        capture = cv2.VideoCapture(str(recorder.path))
        try:
            self.assertTrue(capture.read()[0])
        finally:
            capture.release()
        self.assertEqual(json.loads(recorder.path.with_suffix('.json').read_text())["frames"], 3)

    def test_counts_file(self):
        path = self.path/"counts.json"
        path.write_text(json.dumps([dict(image_id="case.png", expected_counts=ONES)]))
        self.assertEqual(load_expected(str(path), "case.png"), ONES)
        with self.assertRaises(ValueError): load_expected(str(path), "absent")
        for invalid in (["bad"], [dict(image_id="case.png", expected_counts=ONES, boxes=[])],
                        [dict(image_id="case.png", expected_counts=ONES)] * 2):
            path.write_text(json.dumps(invalid))
            with self.assertRaises(ValueError):
                load_expected(str(path), "case.png")

    def test_latency_evidence_boundaries_and_unique_frames(self):
        for mode in ("mock", "real_headless", "real"):
            for n in range(20):
                row = dict(frame_id=n, source="camera:1", mode=mode, model="baseline3", model_version="test-only",
                           width=640, height=480, inference_ms=1, processing_ms=2, processing_fps=10,
                           status="PASS", expected_counts=json.dumps(ONES))
                self.journal.latency(row)
                self.journal.latency(row)  # Repeated presentation is not a fresh frame.
        self.journal.summarize()
        groups = json.loads((self.journal.path/"latency_summary.json").read_text())
        self.assertEqual([g["frames"] for g in groups], [20, 20, 20])
        self.assertEqual([g["evidence"] for g in groups],
                         ["MOCK_DEBUG_ONLY", "INSUFFICIENT_LIVE_EVIDENCE", "LIVE_PROXY_20_FRAMES"])


if __name__ == "__main__":
    unittest.main()
