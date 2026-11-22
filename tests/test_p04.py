"""Regression checks for P04 baseline1.

Run directly:  python tests/test_p04.py
Or with pytest:  pytest tests/test_p04.py -v
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.p04 import catalogue, features, model, pipeline  # noqa: E402


def test_catalogue_ids_and_names() -> None:
    assert catalogue.NON_CATALOGUE_ID == 0
    assert [c["class_id"] for c in catalogue.TARGET_CLASSES] == [1, 2, 3, 4, 5]
    assert catalogue.class_name(0) == "non_catalogue"
    assert catalogue.class_name(1) == "long_screw"
    assert catalogue.class_name(5) == "flat_washer"


def test_hog_feature_shape() -> None:
    config = pipeline.load_config(os.path.join(REPO_ROOT, "config", "baseline1.json"))
    image = np.random.default_rng(0).integers(0, 256, (128, 128, 3), dtype=np.uint8)
    vec = features.extract_hog(image, tuple(config["image_size"]), config["hog"])
    expected_dim = features.hog_vector_length(tuple(config["image_size"]), config["hog"])
    assert vec.shape == (expected_dim,)
    assert vec.dtype == np.float32


def test_counts_validation() -> None:
    from predict import validate_counts  # noqa: WPS433
    validate_counts({"1": 2, "2": 0, "3": 1, "4": 0, "5": 3})
    try:
        validate_counts({"1": -1})
        raise AssertionError("negative count should be rejected")
    except ValueError:
        pass
    try:
        validate_counts({"6": 1})
        raise AssertionError("extra key should be rejected")
    except ValueError:
        pass
    try:
        validate_counts({"1": True})
        raise AssertionError("bool count should be rejected")
    except ValueError:
        pass


def test_train_save_load_predict_roundtrip() -> None:
    rng = np.random.default_rng(42)
    n_per = 40
    # two separable synthetic Gaussian blobs in feature space
    X = np.vstack([rng.normal(0.0, 0.3, (n_per, 8)),
                   rng.normal(1.0, 0.3, (n_per, 8))]).astype(np.float32)
    y = np.array(["PASS"] * n_per + ["FAIL"] * n_per)

    config = pipeline.load_config(os.path.join(REPO_ROOT, "config", "baseline1.json"))
    mlp = model.fit_mlp(X, y, config["mlp"])
    bundle = model.make_bundle(config, mlp, feature_dim=8, config_checksum="test")

    with tempfile.TemporaryDirectory() as tmp:
        bundle_path = os.path.join(tmp, "baseline1.joblib")
        model.save_bundle(bundle, bundle_path)
        loaded = model.load_bundle(bundle_path)

    preds = model.predict(loaded, X[:4])
    assert len(preds) == 4
    for pred in preds:
        assert pred["label"] in catalogue.IMAGE_LABELS
        assert pred["score_type"] == "prob"
        assert 0.0 <= pred["score"] <= 1.0


def _run_all() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(tests)} checks passed.")


if __name__ == "__main__":
    _run_all()
