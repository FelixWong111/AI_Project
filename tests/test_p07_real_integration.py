"""STUB_PASS: deployment contracts only; never train or supply production weights."""
from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import sklearn

from src.camera import Frame
from src.p07_bridge import ImageBackend, UnavailableBackend, image_spec, load_backend, validate_result
from src.p07_ui import InferenceWorker
from p07_stubs import ImageStub

from src.work_orders import reference_counts
CONFIG = json.loads(Path("config/p07.json").read_text())
CONFIG.pop('model_registry',None)  # Provider-contract fixtures intentionally override native artifacts.
CONFIG.setdefault("expected_counts", reference_counts())
FRAME = np.zeros((50, 70, 3), np.uint8)
FIXED = reference_counts()


class DeploymentContractTests(unittest.TestCase):
    def test_every_mode_routes_images_to_real_loaders_or_missing_error(self):
        for mode in ("auto", "real", "mock"):
            for model in ("baseline1", "baseline2"):
                with tempfile.TemporaryDirectory() as tmp:
                    config = copy.deepcopy(CONFIG)
                    if model == "baseline1":
                        config["model_config"]["baseline1_bundle"] = str(Path(tmp)/"absent.joblib")
                    else:
                        config["model_config"]["baseline2_bundle"] = str(Path(tmp)/"absent")
                    with self.assertRaises(FileNotFoundError):
                        load_backend(mode, model, config)

    def test_p05_route_uses_owner_factory_and_reuses_instance(self):
        native = ImageStub("baseline2", FIXED)
        with patch("src.p07_bridge.image_spec", return_value=(None, None, {"device": "cpu"}, [])), \
             patch("src.p05_bridge.create_backend", return_value=native) as factory:
            backend = load_backend("mock", "baseline2", CONFIG)
            for _ in range(3):
                result = backend.infer(FRAME, FIXED)
                validate_result(result, "baseline2", FRAME.shape)
            factory.assert_called_once_with(model_id="baseline2", config={"device": "cpu"})
            self.assertFalse(backend.is_mock)
            self.assertEqual(native.calls, 3)

    def test_counts_skip_first_inference_and_cache_supported_result(self):
        native = ImageStub("baseline1", FIXED)
        backend = ImageBackend("baseline1", native, [])
        other = dict(FIXED, **{"1": 99})
        raw = backend.infer(FRAME, other)
        self.assertEqual(native.calls, 0)
        self.assertEqual(raw["status"], "UNSUPPORTED_COUNTS")
        self.assertIsNone(raw["prediction_score"])
        self.assertIsNone(raw["counts"])
        restored = backend.rejudge(raw, FIXED)
        self.assertEqual(native.calls, 1)
        unsupported = backend.rejudge(restored, other)
        restored_again = backend.rejudge(unsupported, FIXED)
        self.assertEqual(native.calls, 1)
        self.assertEqual(restored_again["predicted_label"], "PASS")

    def test_worker_times_first_prediction_after_unsupported_counts(self):
        native = ImageStub("baseline1", FIXED)
        backend = ImageBackend("baseline1", native, [])
        worker = InferenceWorker()
        try:
            previous = backend.infer(FRAME, dict(FIXED, **{"1": 99}))
            worker.submit(backend, "baseline1", Frame(1, FRAME, time.perf_counter(), 0, "image"), FIXED, "pass", 0, previous)
            deadline = time.perf_counter()+3
            result = None
            while result is None and time.perf_counter() < deadline:
                result = worker.poll()
                time.sleep(.001)
            self.assertIsNotNone(result)
            self.assertFalse(result["rejudge"])
            self.assertEqual(result["result"]["model_id"], "baseline1")
            self.assertEqual(result["result"]["model_version"], "TEST-STUB-ONLY")
            self.assertIsNotNone(result["result"]["inference_time_ms"])
        finally:
            worker.close()

    def test_runtime_artifact_change_blocks_old_instance(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp)/"revision.stub"
            marker.write_text("version1")
            native = ImageStub("baseline1", FIXED)
            backend = ImageBackend("baseline1", native, [marker])
            first = backend.infer(FRAME, FIXED)
            marker.write_text("version2-new-size")
            with self.assertRaisesRegex(RuntimeError, "changed"):
                backend.infer(FRAME, FIXED)
            with self.assertRaisesRegex(RuntimeError, "changed"):
                backend.rejudge(first, FIXED)
            replacement = ImageBackend("baseline1", native, [marker])
            self.assertEqual(replacement.infer(FRAME, FIXED)["status"], "PASS")

    def test_invalid_frames_do_not_reach_native_model(self):
        native = ImageStub("baseline1", FIXED)
        backend = ImageBackend("baseline1", native, [])
        for invalid in (None, np.zeros((0, 2, 3), np.uint8), np.zeros((2, 3)), FRAME.astype(float)):
            with self.assertRaises(ValueError):
                backend.infer(invalid, FIXED)
        self.assertEqual(native.calls, 0)

    def test_p04_reuses_owner_hog_and_classifier_mapping_and_updated_config(self):
        from src.adapters.baseline1 import Baseline1Backend
        from src.models.baseline1 import features
        original = json.loads(Path("config/baseline1.json").read_text())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Opaque file is only a loader-test token; no trained model is serialized.
            marker = root/"opaque.stub"
            marker.write_bytes(b"UNIT ONLY: no model parameters")
            config_path = root/"owner-config.json"
            for orientations in (6, 9):
                config = copy.deepcopy(original)
                config["hog"]["orientations"] = orientations
                config["supported_fixed_counts"] = FIXED
                config["freeze"]["bundle_file"] = str(marker)
                config_path.write_text(json.dumps(config), encoding="utf-8")
                digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
                freeze = dict(config_checksum=digest, bundle_checksum=hashlib.sha256(marker.read_bytes()).hexdigest())
                (root/"freeze.json").write_text(json.dumps(freeze))
                dim = features.hog_vector_length(tuple(config["image_size"]), config["hog"])
                classifier = SimpleNamespace(classes_=np.array(["FAIL", "PASS"]), n_features_in_=dim,
                                             predict_proba=Mock(return_value=np.array([[.9, .1]])))
                bundle = dict(mlp=classifier, classes_=["FAIL", "PASS"], feature_dim=dim, image_size=config["image_size"],
                              hog_params=config["hog"], preprocessing=config["preprocessing"], supported_fixed_counts=FIXED,
                              score_type="prob", class_catalogue=config["class_catalogue"],
                              metadata=dict(sklearn_version=sklearn.__version__, config_checksum=digest))
                with patch("src.models.baseline1.model.load_bundle", return_value=bundle) as loader:
                    backend = Baseline1Backend(config_path, root)
                    prediction = backend.infer(FRAME, FIXED)
                    self.assertEqual(prediction["predicted_label"], "FAIL")
                    self.assertEqual(prediction["prediction_score"], .9)
                    expected = features.extract_hog(FRAME, tuple(config["image_size"]), config["hog"])
                    np.testing.assert_array_equal(classifier.predict_proba.call_args.args[0][0], expected)
                    loader.assert_called_once_with(str(marker))
                invalid_bundle = copy.deepcopy(bundle)
                invalid_bundle["hog_params"]["orientations"] += 1
                with patch("src.models.baseline1.model.load_bundle", return_value=invalid_bundle):
                    with self.assertRaisesRegex(ValueError, "HOG"):
                        Baseline1Backend(config_path, root)
                freeze["bundle_checksum"] = "wrong"
                (root/"freeze.json").write_text(json.dumps(freeze))
                with self.assertRaisesRegex(ValueError, "SHA256"):
                    Baseline1Backend(config_path, root)

    def test_unavailable_model_has_no_predictions(self):
        for model in ("baseline1", "baseline2"):
            backend = UnavailableBackend(model, "specific missing file")
            result = backend.infer()
            self.assertFalse(backend.is_mock)
            self.assertEqual(result["status"], "ERROR")
            self.assertIsNone(result["prediction_score"])
            self.assertIsNone(result["predicted_label"])
            self.assertIsNone(result["counts"])
            self.assertEqual(result["objects"], [])

    def test_cross_model_results_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "model_id"):
            validate_result(dict(model_id="baseline2", status="PASS", predicted_label="PASS"), "baseline1", FRAME.shape)


if __name__ == "__main__":
    unittest.main()
