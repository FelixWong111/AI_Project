"""Image-level evaluation metrics and latency measurement for baseline1."""

from __future__ import annotations

import time
from typing import Callable, Dict, List

import numpy as np

from .catalogue import LABEL_FAIL, LABEL_PASS


def image_level_metrics(y_true: List[str], y_pred: List[str]) -> Dict:
    """Confusion counts and P/R/F1 for the FAIL class (positive class)."""
    tp = fp = fn = tn = 0
    for true, pred in zip(y_true, y_pred):
        if true == LABEL_FAIL and pred == LABEL_FAIL:
            tp += 1
        elif true == LABEL_PASS and pred == LABEL_FAIL:
            fp += 1
        elif true == LABEL_FAIL and pred == LABEL_PASS:
            fn += 1
        elif true == LABEL_PASS and pred == LABEL_PASS:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    recall_pass = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced_accuracy = (recall + recall_pass) / 2

    return {
        "positive_class": LABEL_FAIL,
        "confusion": {"TP": tp, "FP": fp, "FN": fn, "TN": tn},
        "precision_fail": round(precision, 4),
        "recall_fail": round(recall, 4),
        "f1_fail": round(f1, 4),
        "balanced_accuracy": round(balanced_accuracy, 4),
        "support": {"FAIL": tp + fn, "PASS": tn + fp},
    }


def measure_latency(predict_fn: Callable[[], None], n_frames: int = 20) -> Dict:
    """Measure per-call latency in ms; return median and 95th percentile.

    ``predict_fn`` runs one full inference call (preprocessing + inference +
    postprocessing). Timing excludes camera acquisition, matching the
    project's offline batch timing boundary.
    """
    latencies_ms: List[float] = []
    for _ in range(max(n_frames, 1)):
        start = time.perf_counter()
        predict_fn()
        latencies_ms.append((time.perf_counter() - start) * 1000.0)
    arr = np.asarray(latencies_ms)
    return {
        "n_frames": int(arr.size),
        "latency_ms_median": round(float(np.median(arr)), 3),
        "latency_ms_p95": round(float(np.percentile(arr, 95)), 3),
        "latency_ms_mean": round(float(arr.mean()), 3),
    }
