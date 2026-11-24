"""P07 integration only: delegate HOG and MLP to P04's unchanged functions."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from src.work_orders import reference_counts, validate_counts, unsupported_counts


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Baseline1Backend:
    is_mock = False

    def __init__(self, config_path, repo_root, bundle_override=None):
        from src.p07_bridge import validate_counts
        self.config_path = Path(config_path)
        config = json.loads(self.config_path.read_text(encoding="utf-8-sig"))
        declared = bundle_override or config.get("freeze", {}).get("bundle_file")
        if not declared:
            raise ValueError("P04 config.freeze.bundle_file is missing")
        path = Path(declared)
        self.path = path if path.is_absolute() else Path(repo_root) / path
        self.freeze_path = self.path.parent / "freeze.json"
        missing = [str(p) for p in (self.path, self.freeze_path) if not p.is_file()]
        if missing:
            raise FileNotFoundError("P04 missing: " + ", ".join(missing))
        freeze = json.loads(self.freeze_path.read_text(encoding="utf-8-sig"))
        version = digest(self.path)
        if freeze.get("bundle_checksum") != version:
            raise ValueError("P04 bundle SHA256 differs from freeze.json; replace the matching delivery set")
        if freeze.get("config_checksum") != digest(self.config_path):
            raise ValueError("P04 configuration differs from freeze.json; reload a matching config/bundle")
        from src.models.baseline1 import features, model
        import sklearn
        self.features, self.model = features, model
        self.bundle = model.load_bundle(str(self.path))
        if self.bundle["metadata"].get("sklearn_version") != sklearn.__version__:
            raise ValueError("P04 sklearn version differs from deployed environment")
        if self.bundle["metadata"].get("config_checksum") != digest(self.config_path):
            raise ValueError("P04 bundle references a different configuration")
        self.fixed = validate_counts(self.bundle["supported_fixed_counts"])
        if self.fixed != reference_counts():
            raise ValueError('Trained baseline1 counts differ from the common reference; config cannot relabel a model')
        if (self.fixed != validate_counts(config["supported_fixed_counts"])
                or tuple(self.bundle["image_size"]) != tuple(config["image_size"])
                or self.bundle["hog_params"] != config["hog"]
                or self.bundle["preprocessing"] != config["preprocessing"]
                or self.bundle["score_type"] != config["decision"]["score_type"]):
            raise ValueError("P04 bundle preprocessing/HOG/fixed counts do not match config")
        classes = [str(c) for c in self.bundle["mlp"].classes_]
        if classes != self.bundle["classes_"] or set(classes) != {"PASS", "FAIL"}:
            raise ValueError("P04 class mapping must explicitly match PASS/FAIL")
        dim = features.hog_vector_length(tuple(self.bundle["image_size"]), self.bundle["hog_params"])
        if self.bundle["feature_dim"] != dim or self.bundle["mlp"].n_features_in_ != dim:
            raise ValueError("P04 feature dimension mismatch")
        self.metadata = dict(model="baseline1", model_version=version, deployment_stage="candidate",
                             output_capability="image_level", supported_fixed_counts=self.fixed,
                             preprocessing=dict(self.bundle["preprocessing"], image_size=self.bundle["image_size"], hog=self.bundle["hog_params"]),
                             score_convention="P(predicted label), uncalibrated MLP probability",
                             class_catalogue=self.bundle["class_catalogue"], coordinate_convention=None,
                             thresholds={"decision": "argmax in classifier.classes_ order"},
                             rejection=None, size_tolerance=None, calibration=None,
                             model_file_size_bytes=self.path.stat().st_size, bundle_file=str(self.path),
                             config_file=str(self.config_path), training_metadata=self.bundle["metadata"])

    def synchronize(self):
        pass

    def infer(self, frame_bgr, expected_counts):
        if validate_counts(expected_counts) != self.fixed:
            return unsupported_counts()
        vector = self.features.extract_hog(frame_bgr, tuple(self.bundle["image_size"]), self.bundle["hog_params"])
        prediction = self.model.predict(self.bundle, vector.reshape(1, -1))[0]
        if getattr(self, 'scene_guard', None) is not None:
            standardized = self.bundle['mlp'].named_steps['standardscaler'].transform(vector.reshape(1, -1))
            supported, distance = self.scene_guard.check(standardized)
            if not supported:
                return dict(status='REVIEW', predicted_label=None, prediction_score=None, score_type='none',
                            objects=[], counts=None, non_catalogue_count=None,
                            raw_baseline_prediction=prediction, scene_distance=distance,
                            message='Outside baseline1 training support: REVIEW. Use the supported tabletop setup.')
        return dict(status=prediction["label"], predicted_label=prediction["label"],
                    prediction_score=prediction["score"], score_type=prediction["score_type"],
                    objects=[], counts=None, non_catalogue_count=None)
