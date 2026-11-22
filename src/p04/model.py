"""MLP model bundle: training, persistence and prediction for baseline1."""

from __future__ import annotations

import platform
from datetime import datetime, timezone
from typing import Dict, List, Optional

import joblib
import numpy as np
import sklearn
from sklearn.neural_network import MLPClassifier

from . import catalogue


def balanced_sample_weights(y: np.ndarray) -> np.ndarray:
    """Compute balanced per-sample weights (n / (n_classes * count_c))."""
    classes, counts = np.unique(y, return_counts=True)
    n_total = len(y)
    weight_map = {
        c: n_total / (len(classes) * count)
        for c, count in zip(classes, counts)
    }
    return np.array([weight_map[label] for label in y], dtype=np.float64)


def fit_mlp(X: np.ndarray, y: np.ndarray, mlp_params: Dict) -> MLPClassifier:
    """Fit the scikit-learn MLP with balanced sample weights."""
    params = dict(mlp_params)
    random_state = params.pop("random_state", 42)
    class_weight = params.pop("class_weight", "balanced")
    mlp = MLPClassifier(random_state=random_state, **params)
    sample_weight = balanced_sample_weights(y) if class_weight == "balanced" else None
    mlp.fit(X, y, sample_weight=sample_weight)
    return mlp


def make_bundle(config: Dict, mlp: MLPClassifier, feature_dim: int,
                config_checksum: Optional[str] = None,
                extra_meta: Optional[Dict] = None) -> Dict:
    """Assemble the frozen bundle persisted to disk."""
    return {
        "model_id": config["model_id"],
        "method": config["method"],
        "class_catalogue": config.get("class_catalogue", []),
        "mlp": mlp,
        "classes_": [str(c) for c in mlp.classes_],
        "image_size": tuple(config["image_size"]),
        "hog_params": config["hog"],
        "preprocessing": config.get("preprocessing", {}),
        "supported_fixed_counts": dict(config["supported_fixed_counts"]),
        "score_type": config["decision"]["score_type"],
        "feature_dim": feature_dim,
        "metadata": {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "sklearn_version": sklearn.__version__,
            "numpy_version": np.__version__,
            "platform": platform.platform(),
            "random_state": config["mlp"].get("random_state", 42),
            "config_checksum": config_checksum,
            **(extra_meta or {}),
        },
    }


def save_bundle(bundle: Dict, path: str) -> None:
    joblib.dump(bundle, path)


def load_bundle(path: str) -> Dict:
    bundle = joblib.load(path)
    if bundle.get("model_id") != catalogue.MODEL_ID:
        raise ValueError(f"Bundle model_id mismatch: {bundle.get('model_id')}")
    return bundle


def predict_proba(bundle: Dict, X: np.ndarray) -> tuple[np.ndarray, List[str]]:
    """Return class probabilities aligned with ``bundle['classes_']``."""
    proba = bundle["mlp"].predict_proba(X)
    return proba, bundle["classes_"]


def predict(bundle: Dict, X: np.ndarray) -> List[Dict]:
    """Predict labels with the probability of the predicted class as score."""
    proba, classes = predict_proba(bundle, X)
    results: List[Dict] = []
    for row in proba:
        idx = int(np.argmax(row))
        results.append({
            "label": classes[idx],
            "score": float(row[idx]),
            "score_type": bundle["score_type"],
            "proba": {classes[i]: float(p) for i, p in enumerate(row)},
        })
    return results
