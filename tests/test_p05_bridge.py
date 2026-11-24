"""Regression checks for P05/P07 handoff and count-only restoration."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from src.adapters.baseline2 import WEIGHT_FILE
from src.p05_bridge import create_backend
from src.p07_bridge import load_backend, validate_result
from src.p07_output import export_result

FIXED = dict.fromkeys("12345", 1)
OTHER = FIXED | {"1": 0}
FRAME = np.zeros((40, 60, 3), dtype=np.uint8)


class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.bundle = Path(self.directory.name)
        for name in ("baseline2.json", WEIGHT_FILE):
            (self.bundle / name).write_bytes(b"test-only")
        self.adapter = Mock()
        self.adapter.manifest = {"model_version": "test-only"}
        self.adapter.fixed_counts = FIXED
        self.adapter.features = SimpleNamespace(device=SimpleNamespace(type="cpu"))
        self.adapter.run_config.return_value = {
            "model_version": "test-only", "output_capability": "image_level",
            "supported_fixed_counts": FIXED,
        }
        self.adapter.predict_frame.return_value = {
            "image_id": "frame.png", "model_id": "baseline2", "status": "PASS",
            "predicted_label": "PASS", "prediction_score": .2, "score_type": "prob",
            "non_catalogue_count": None,
        }
        replacement = patch("src.p05_bridge.Baseline2", return_value=self.adapter)
        replacement.start()
        self.addCleanup(replacement.stop)
        self.config = {
            "bridge": "src.p05_bridge:create_backend",
            "model_config": {"baseline2_bundle": str(self.bundle), "device": "cpu"},
        }
        self.backend = load_backend("real", "baseline2", self.config)

    def test_load_protocol_and_metadata(self):
        self.assertFalse(self.backend.is_mock)
        self.assertEqual(self.backend.metadata["model_file_size_bytes"], 18)
        self.backend.synchronize()
        validate_result(self.backend.infer(FRAME, FIXED), "baseline2", FRAME.shape)

    def test_unsupported_first_can_restore_without_inference(self):
        raw = self.backend.infer(FRAME, OTHER)
        self.assertEqual(raw["status"], "UNSUPPORTED_COUNTS")
        self.assertIsNone(raw["prediction_score"])
        restored = self.backend.rejudge(raw, FIXED)
        self.assertEqual(restored["status"], "PASS")
        self.assertEqual(restored["prediction_score"], .2)
        self.assertEqual(self.adapter.predict_frame.call_count, 1)
        self.assertEqual(self.adapter.predict_frame.call_args.args[1], FIXED)

    def test_cached_results_are_bound_to_their_own_frame(self):
        first = self.backend.infer(FRAME, FIXED)
        self.adapter.predict_frame.return_value = {
            **self.adapter.predict_frame.return_value, "status": "FAIL",
            "predicted_label": "FAIL", "prediction_score": .8,
        }
        second = self.backend.infer(FRAME, FIXED)
        for observation, expected_label in ((first, "PASS"), (second, "FAIL")):
            unsupported = self.backend.rejudge(observation, OTHER)
            restored = self.backend.rejudge(unsupported, FIXED)
            self.assertEqual(restored["predicted_label"], expected_label)
            self.assertIsNone(restored["counts"])
            self.assertEqual(restored["objects"], [])
            validate_result(restored, "baseline2", FRAME.shape)
        self.assertEqual(self.adapter.predict_frame.call_count, 2)

    def test_source_and_cached_results_are_immutable(self):
        original = FRAME.copy()
        prediction = self.backend.infer(FRAME, FIXED)
        snapshot = json.dumps(prediction, sort_keys=True)
        self.backend.rejudge(prediction, OTHER)
        self.assertTrue(np.array_equal(original, FRAME))
        self.assertEqual(json.dumps(prediction, sort_keys=True), snapshot)

    def test_factory_and_validation_errors(self):
        for model in ("baseline1", "baseline3", "final"):
            with self.assertRaises(ValueError):
                create_backend(model_id=model, config={})
        with self.assertRaises(ValueError):
            create_backend(model_id="baseline2", config={})
        with self.assertRaises(ValueError):
            create_backend(model_id="baseline2", config={
                "baseline2_bundle": str(self.bundle), "expected_model_version": "wrong",
            })
        for invalid in ({}, FIXED | {"1": True}):
            with self.assertRaises(ValueError):
                self.backend.infer(FRAME, invalid)
        self.assertEqual(self.adapter.predict_frame.call_count, 0)

    def test_gpu_synchronization(self):
        device = SimpleNamespace(type="cuda")
        synchronize = Mock()
        self.adapter.features = SimpleNamespace(
            device=device, torch=SimpleNamespace(cuda=SimpleNamespace(synchronize=synchronize))
        )
        self.backend.synchronize()
        synchronize.assert_called_once_with(device)

    def test_preserve_worker_error(self):
        error = dict(status="ERROR", predicted_label=None, message="inference failed")
        self.assertEqual(self.backend.rejudge(error, FIXED), error)

    def test_p07_export_contains_only_image_outputs(self):
        result = self.backend.infer(FRAME, FIXED)
        output = self.bundle / "export"
        export_result(output, "sample.png", "baseline2", result, FIXED, self.backend.metadata)
        self.assertEqual({p.name for p in output.iterdir()}, {"images.csv", "run_config.json"})
        self.assertNotIn("_p05_prediction", (output / "images.csv").read_text())
        config = json.loads((output / "run_config.json").read_text())
        self.assertEqual(config["per_image_expected_counts"], {"sample.png": FIXED})


if __name__ == "__main__":
    unittest.main()
