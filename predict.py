"""Unified batch inference entry for the AI Project (team contract §5.1).

Command (project contract)
--------------------------
    python predict.py --input_dir TEST_FOLDER --model MODEL_ID \
        --output_dir RESULTS --counts_file expected_counts.json

MODEL_ID
--------
- ``baseline1`` : P04 HOG + scikit-learn MLP, image-level baseline (implemented here)
- ``baseline2`` : P05 frozen CNN features + logistic regression (stub; owner P05)
- ``baseline3`` : P06 object detector (stub; owner P06)
- ``final``     : P06 improved final system (stub; owner P06)

Outputs (contract §5.3, fixed column order)
------------------------------------------
``images.csv``: image_id, model_id, status, predicted_label,
    prediction_score, score_type, non_catalogue_count
``objects.csv`` / ``counts.csv``: object-capable systems only. The image-only
    baseline1 is exempt and must NOT fabricate those outputs.

Status enum (§5.3, pending group confirmation)
---------------------------------------------
PASS | FAIL | REVIEW | UNSUPPORTED_COUNTS | ERROR
- PASS/FAIL: produced a decision; ``status`` equals ``predicted_label``.
- UNSUPPORTED_COUNTS: expected-count vector not supported by this image-only
  model; predicted_label and prediction_score are empty.
- ERROR: image could not be loaded/decoded or inference failed.
- REVIEW: reserved for unjudgeable cases; baseline1 does not emit it.

score_type enum: prob | logit | none | empty.
baseline1 uses ``prob`` (MLP softmax probability of the predicted class,
uncalibrated).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
from typing import Dict, List, Optional

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

from src.p04 import dataset, features, model, pipeline  # noqa: E402

SUPPORTED_MODELS = {"baseline1", "baseline2", "baseline3", "final"}
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")
COUNT_KEYS = {"1", "2", "3", "4", "5"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified batch inference interface")
    p.add_argument("--input_dir", required=True, help="folder with test images")
    p.add_argument("--model", required=True, help="model ID: baseline1/baseline2/baseline3/final")
    p.add_argument("--output_dir", required=True, help="folder for prediction outputs")
    p.add_argument("--counts_file", default=None, help="expected_counts.json")
    p.add_argument("--bundle", default=None, help="override baseline1 bundle path")
    return p.parse_args()


def normalize_counts(counts: Dict) -> Dict[str, int]:
    return {str(k): int(v) for k, v in counts.items()}


def validate_counts(counts: Dict[str, int]) -> None:
    """Reject bool/float/negative/extra keys per contract §5.2 / README §6."""
    for key, value in counts.items():
        if key not in COUNT_KEYS:
            raise ValueError(f"invalid expected_counts key: {key}")
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"expected_counts[{key}] must be a non-negative integer, got {value!r}")


def load_cases(counts_file: Optional[str]) -> List[Dict]:
    if not counts_file:
        return []
    with open(counts_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]
    cases: List[Dict] = []
    for entry in data:
        raw_counts = entry.get("expected_counts", {})
        validate_counts(raw_counts)
        expected = normalize_counts(raw_counts)
        cases.append({"image_id": entry["image_id"], "expected_counts": expected})
    return cases


def list_input_images(input_dir: str) -> List[str]:
    names = [n for n in os.listdir(input_dir)
             if os.path.splitext(n)[1].lower() in IMAGE_SUFFIXES]
    return sorted(names)


def _cv2_version() -> str:
    try:
        import cv2
        return cv2.__version__
    except Exception:  # noqa: BLE001
        return "unknown"


def bundle_default_path() -> str:
    return os.path.join(REPO_ROOT, "models", "baseline1", "baseline1.joblib")


def _empty_row(image_id: str, model_id: str, status: str) -> Dict:
    return {
        "image_id": image_id,
        "model_id": model_id,
        "status": status,
        "predicted_label": None,
        "prediction_score": None,
        "score_type": None,
        "non_catalogue_count": None,
    }


def run_baseline1(args: argparse.Namespace, cases: List[Dict]) -> Dict:
    bundle_path = args.bundle or bundle_default_path()
    if not os.path.isfile(bundle_path):
        raise SystemExit(f"baseline1 bundle not found: {bundle_path}. "
                         f"Train it first with tools/train_baseline1.py")
    bundle = model.load_bundle(bundle_path)
    supported = bundle["supported_fixed_counts"]

    work: List[Dict] = []
    if cases:
        for case in cases:
            work.append({"image_id": case["image_id"], "expected_counts": case["expected_counts"]})
    else:
        for name in list_input_images(args.input_dir):
            work.append({"image_id": name, "expected_counts": dict(supported)})

    rows: List[Dict] = []
    for item in work:
        image_id = item["image_id"]
        expected = item["expected_counts"]
        row = _empty_row(image_id, bundle["model_id"], "ERROR")

        # Image-only baseline supports one fixed expected-count vector.
        if expected and expected != supported:
            row["status"] = "UNSUPPORTED_COUNTS"
            rows.append(row)
            continue

        image_path = dataset.find_image_path(args.input_dir, image_id)
        if image_path is None or not os.path.isfile(image_path):
            row["status"] = "ERROR"
            rows.append(row)
            continue

        try:
            bgr = features.load_image_bgr(image_path)
            feat = features.extract_hog(bgr, tuple(bundle["image_size"]), bundle["hog_params"])
            preds = model.predict(bundle, feat.reshape(1, -1))
            pred = preds[0]
            row["status"] = pred["label"]  # PASS / FAIL
            row["predicted_label"] = pred["label"]
            row["prediction_score"] = round(pred["score"], 6)
            row["score_type"] = pred["score_type"]
        except Exception as exc:  # noqa: BLE001 - failed inputs recorded explicitly
            row["status"] = "ERROR"
            print(f"[predict] ERROR on {image_id}: {exc}", file=sys.stderr)
        rows.append(row)

    run_config = {
        "model": bundle["model_id"],
        "model_version": {
            "trained_at": bundle["metadata"]["trained_at"],
            "config_checksum": bundle["metadata"].get("config_checksum"),
            "bundle_checksum": pipeline.sha256_file(bundle_path),
            "run_id": "TBD (run_id convention pending group confirmation)",
        },
        "output_capability": "image_level",
        "class_catalogue": bundle["class_catalogue"],
        "thresholds": {"decision_score_threshold": 0.5},
        "preprocessing": bundle["preprocessing"],
        "coordinate_convention": "N/A for image-level baseline (no boxes)",
        "score_convention": {
            "score_type": bundle["score_type"],
            "meaning": "MLP softmax probability of the predicted class; uncalibrated model probability, calibration not claimed.",
        },
        "hardware": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "sklearn": bundle["metadata"]["sklearn_version"],
            "numpy": bundle["metadata"]["numpy_version"],
            "opencv": _cv2_version(),
        },
        "rejection": {
            "supported": False,
            "note": "Image-level baseline1 does not localize/reject non-catalogue objects; ID 0 rejection belongs to baseline3/final.",
        },
        "size_tolerance": "N/A for image-level baseline; belongs to baseline3/final and rig calibration.",
        "calibration": "N/A; scale/calibration belongs to P01/P06.",
        "per_image_expected_counts": {item["image_id"]: item["expected_counts"] for item in work},
        "supported_fixed_counts": supported,
        "bundle_file": os.path.relpath(bundle_path, REPO_ROOT),
        "metadata": bundle["metadata"],
    }
    return {"rows": rows, "run_config": run_config}


def run_stub(model_id: str, cases: List[Dict], input_dir: str) -> Dict:
    ids = [c["image_id"] for c in cases] if cases else list_input_images(input_dir)
    rows = [_empty_row(image_id, model_id, "ERROR") for image_id in ids]
    run_config = {
        "model": model_id,
        "output_capability": None,
        "note": "Stub in P04 entry. This model ID is owned by another team member "
                "(baseline2=P05, baseline3/final=P06); implementation and weights are not provided here.",
    }
    return {"rows": rows, "run_config": run_config}


def write_outputs(output_dir: str, result: Dict) -> None:
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "images.csv")
    fieldnames = ["image_id", "model_id", "status", "predicted_label",
                  "prediction_score", "score_type", "non_catalogue_count"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result["rows"])

    with open(os.path.join(output_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(result["run_config"], f, indent=2, ensure_ascii=False)


def main() -> int:
    args = parse_args()

    if args.model not in SUPPORTED_MODELS:
        raise SystemExit(f"Unknown MODEL_ID: {args.model}. "
                         f"Supported: {sorted(SUPPORTED_MODELS)}")

    try:
        cases = load_cases(args.counts_file)
    except (ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Invalid counts_file: {exc}")

    if args.model == "baseline1":
        result = run_baseline1(args, cases)
    else:
        result = run_stub(args.model, cases, args.input_dir)

    write_outputs(args.output_dir, result)
    n_ok = sum(1 for r in result["rows"] if r["status"] in ("PASS", "FAIL"))
    print(f"[predict] model={args.model} rows={len(result['rows'])} decided={n_ok} "
          f"-> {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
