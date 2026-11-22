"""Train and freeze baseline1 (HOG + scikit-learn MLP, image-level).

Usage
-----
    python tools/train_baseline1.py \
        --config config/baseline1.json \
        --gt-csv data/ground_truth_images.csv \
        --images-dir data/images \
        --output-dir models/baseline1

Outputs (under --output-dir)
----------------------------
baseline1.joblib        frozen model bundle (MLP + params + metadata)
config.frozen.json      copy of the frozen configuration
metrics.json            validation metrics, latency and support
samples.json            train/val image lists used for this run
freeze.json             freeze timestamp, checksum and bundle reference
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys

# Make the repo root importable when run as a script.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.p04 import dataset, evaluate, model, pipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train and freeze baseline1 HOG+MLP")
    p.add_argument("--config", default=os.path.join(REPO_ROOT, "config", "baseline1.json"))
    p.add_argument("--gt-csv", required=True, help="ground_truth_images.csv path")
    p.add_argument("--images-dir", required=True, help="directory with images")
    p.add_argument("--output-dir", default=os.path.join(REPO_ROOT, "models", "baseline1"))
    p.add_argument("--split-train", default="train")
    p.add_argument("--split-val", default="val")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    config = pipeline.load_config(args.config)
    pipeline.ensure_output_dir(args.output_dir)

    gt = dataset.load_ground_truth(args.gt_csv)
    train_samples = dataset.build_samples(gt, args.images_dir, args.split_train)
    val_samples = dataset.build_samples(gt, args.images_dir, args.split_val)

    if not train_samples:
        raise SystemExit("No training samples found. Check --gt-csv and --images-dir.")

    image_size = tuple(config["image_size"])
    hog_params = config["hog"]

    X_train, y_train, _ = pipeline.extract_features(train_samples, image_size, hog_params)
    print(f"[train] train samples={len(train_samples)} feature_dim={X_train.shape[1]}")

    mlp = model.fit_mlp(X_train, y_train, config["mlp"])

    config_checksum = pipeline.sha256_file(args.config)
    bundle = model.make_bundle(
        config, mlp, feature_dim=X_train.shape[1],
        config_checksum=config_checksum,
        extra_meta={"train_samples": len(train_samples), "val_samples": len(val_samples)},
    )

    metrics: dict = {"train": {"n_samples": len(train_samples)}, "validation": {}}
    y_pred_val: list[str] = []
    y_true_val: list[str] = []

    if val_samples:
        X_val, y_val, val_ids = pipeline.extract_features(val_samples, image_size, hog_params)
        preds = model.predict(bundle, X_val)
        y_true_val = list(y_val)
        y_pred_val = [p["label"] for p in preds]
        metrics["validation"] = evaluate.image_level_metrics(y_true_val, y_pred_val)

        # Latency: full preprocessing + inference on validation frames.
        def _predict_one() -> None:
            sample = val_samples[0]
            bgr = pipeline.features.load_image_bgr(sample["path"])
            feat = pipeline.features.extract_hog(bgr, image_size, hog_params)
            model.predict(bundle, feat.reshape(1, -1))

        latency = evaluate.measure_latency(_predict_one, n_frames=max(20, len(val_samples)))
        metrics["validation"]["latency"] = latency
        metrics["validation"]["val_image_ids"] = val_ids
    else:
        print("[train] WARNING: no validation split; validation metrics skipped.")

    bundle_path = os.path.join(args.output_dir, "baseline1.joblib")
    model.save_bundle(bundle, bundle_path)

    frozen_config_path = os.path.join(args.output_dir, "config.frozen.json")
    with open(frozen_config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    samples_path = os.path.join(args.output_dir, "samples.json")
    with open(samples_path, "w", encoding="utf-8") as f:
        json.dump({
            "train": [s["image_id"] for s in train_samples],
            "val": [s["image_id"] for s in val_samples],
        }, f, indent=2)

    freeze = {
        "model_id": config["model_id"],
        "frozen": True,
        "frozen_at": bundle["metadata"]["trained_at"],
        "bundle_file": os.path.relpath(bundle_path, REPO_ROOT),
        "config_checksum": config_checksum,
        "bundle_checksum": pipeline.sha256_file(bundle_path),
        "sklearn_version": bundle["metadata"]["sklearn_version"],
        "platform": platform.platform(),
        "training_command": " ".join(sys.argv),
        "python": sys.version.split()[0],
    }
    freeze_path = os.path.join(args.output_dir, "freeze.json")
    with open(freeze_path, "w", encoding="utf-8") as f:
        json.dump(freeze, f, indent=2, ensure_ascii=False)

    print("[train] frozen bundle:", bundle_path)
    if metrics["validation"]:
        print("[train] validation metrics:", json.dumps(metrics["validation"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
