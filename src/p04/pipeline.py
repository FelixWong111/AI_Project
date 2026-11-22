"""Shared helpers: config loading, feature extraction, checksums."""

from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, List, Tuple

import numpy as np

from . import features


def load_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_features(samples: List[Dict], image_size: Tuple[int, int],
                     hog_params: Dict) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Extract HOG features for a sample list.

    Returns X (n, dim), y (n,) labels, image_ids (n,).
    """
    X: List[np.ndarray] = []
    y: List[str] = []
    ids: List[str] = []
    for sample in samples:
        bgr = features.load_image_bgr(sample["path"])
        X.append(features.extract_hog(bgr, image_size, hog_params))
        y.append(sample["label"])
        ids.append(sample["image_id"])
    return np.asarray(X, dtype=np.float32), np.asarray(y), ids


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
