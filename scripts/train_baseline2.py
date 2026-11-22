"""Train SW-02 on P03's scene-grouped train/validation manifests."""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory

import numpy as np
from PIL import __version__ as pillow_version
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.adapters.baseline2 import (
    COUNT_KEYS, IMAGE_FIELDS, MODEL_ID, PREPROCESSING, WEIGHT_FILE, WEIGHT_SHA256,
    ResNet18Features, image_filename, model_version, read_rgb, sha256, validate_counts,
)

SEED = 42
FAIL_THRESHOLD = 0.5
VALID_STATUS = {"OK", "VALID", "VERIFIED", "COMPLETE"}
SKIP_STATUS = {"REVIEW", "UNJUDGEABLE", "IGNORE", "IGNORED", "ERROR"}
SPLITS = {"train": "train", "val": "val", "validation": "val", "test": "test"}


def csv_rows(path, fields):
    with path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        if not set(fields).issubset(reader.fieldnames or ()):
            raise ValueError(f"{path} must contain {', '.join(fields)}")
        yield from reader


def load_development_rows(args, counts):
    rows = {"train": [], "val": []}
    split_by_scene, image_ids, judgeable = {}, set(), {}
    excluded = Counter()
    for row in csv_rows(args.images_csv, ("image_id", "scene_id", "split", "annotation_status")):
        image_id = image_filename(row["image_id"])
        scene = row["scene_id"].strip()
        raw_split = row["split"].strip().lower()
        split = SPLITS.get(raw_split)
        if not scene or split is None or image_id in image_ids:
            raise ValueError(f"Empty scene, unsupported split or duplicate image_id: {image_id}")
        image_ids.add(image_id)
        if split_by_scene.setdefault(scene, split) != split:
            raise ValueError(f"Scene {scene} occurs in more than one split")
        if split == "test":
            continue
        status = row["annotation_status"].strip().upper()
        if status in SKIP_STATUS:
            excluded[f"{split}:{status}"] += 1
            continue
        if status not in VALID_STATUS:
            raise ValueError(f"{image_id}: unmapped annotation_status {status!r}")
        path = args.image_dir / image_id
        if not path.is_file():
            path = args.image_dir / raw_split / image_id
        if not path.is_file():
            raise FileNotFoundError(f"Development image missing: {image_id}")
        record = {"image_id": image_id, "scene_id": scene, "split": split, "path": path}
        rows[split].append(record)
        judgeable[image_id] = record

    actual = {image_id: Counter() for image_id in judgeable}
    seen_objects = set()
    for row in csv_rows(args.objects_csv, ("image_id", "object_id", "class_id", "ignore")):
        image_id = row["image_id"]
        if image_id not in image_ids:
            raise ValueError(f"Object row refers to an unknown image: {image_id}")
        if image_id not in judgeable:
            continue
        object_id = row["object_id"].strip()
        identity = (image_id, object_id)
        cls = row["class_id"].strip()
        ignore = row["ignore"].strip().lower()
        if (not object_id or identity in seen_objects or cls not in ("0", *COUNT_KEYS)
                or ignore not in ("0", "1", "true", "false")):
            raise ValueError(f"Invalid or duplicate object row: {identity}")
        seen_objects.add(identity)
        if ignore in ("0", "false"):
            actual[image_id][cls] += 1

    for split, records in rows.items():
        records.sort(key=lambda row: row["image_id"])
        for record in records:
            found = actual[record["image_id"]]
            record["label"] = int(found["0"] > 0 or any(found[k] != counts[k] for k in COUNT_KEYS))
        if {row["label"] for row in records} != {0, 1}:
            raise ValueError(f"{split} requires both judgeable PASS and FAIL images")
    return rows, dict(excluded)


def features_for(records, extractor, batch_size):
    return np.concatenate([
        extractor.extract([read_rgb(row["path"]) for row in records[start:start + batch_size]])
        for start in range(0, len(records), batch_size)
    ])


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def train(args):
    import torch
    import torchvision

    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    counts = validate_counts(dict(zip(COUNT_KEYS, args.expected_counts)))
    rows, excluded = load_development_rows(args, counts)
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")
    torch.manual_seed(SEED)
    if args.weights is None:
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        torch.hub.load_state_dict_from_url(weights.url, check_hash=True, map_location="cpu")
        checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / WEIGHT_FILE
    else:
        checkpoint = args.weights
    weight_digest = sha256(checkpoint)
    if weight_digest != WEIGHT_SHA256:
        raise ValueError("Expected the official ResNet18 IMAGENET1K_V1 checkpoint (f37072fd)")

    extractor = ResNet18Features(checkpoint, args.device)
    x_train = features_for(rows["train"], extractor, args.batch_size)
    y_train = np.array([row["label"] for row in rows["train"]])
    x_val = features_for(rows["val"], extractor, args.batch_size)
    y_val = np.array([row["label"] for row in rows["val"]])

    pipeline = make_pipeline(
        StandardScaler(), LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=SEED)
    )
    pipeline.fit(x_train, y_train)
    scaler, classifier = pipeline.steps[0][1], pipeline.steps[1][1]
    scores = pipeline.predict_proba(x_val)[:, list(classifier.classes_).index(1)]
    predicted = (scores >= FAIL_THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_val, predicted, labels=[0, 1]).ravel()
    p, r, f1, _ = precision_recall_fscore_support(
        y_val, predicted, average="binary", pos_label=1, zero_division=0
    )
    input_rows = [
        {"image_id": row["image_id"], "scene_id": row["scene_id"], "split": split,
         "label": ("PASS", "FAIL")[row["label"]], "sha256": sha256(row["path"])}
        for split, records in rows.items() for row in records
    ]
    input_digest = hashlib.sha256(
        json.dumps(input_rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    manifest = {
        "schema": "baseline2/v1",
        "supported_fixed_counts": counts,
        "preprocessing": PREPROCESSING,
        "backbone": "torchvision ResNet18 IMAGENET1K_V1",
        "backbone_sha256": weight_digest,
        "development_set_sha256": input_digest,
        "training": {"seed": SEED, "batch_size": args.batch_size, "device": args.device,
                     "C": 1.0, "solver": "lbfgs", "max_iter": 2000},
        "versions": {"python": sys.version.split()[0], "torch": torch.__version__,
                     "torchvision": torchvision.__version__, "scikit_learn": sklearn.__version__,
                     "numpy": np.__version__, "pillow": pillow_version},
        "classifier": {
            "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
            "coef": classifier.coef_[0].tolist(), "intercept": float(classifier.intercept_[0]),
            "fail_probability_threshold": FAIL_THRESHOLD,
        },
    }
    manifest["model_version"] = model_version(manifest)
    validation = {
        "partition": "validation", "positive_label": "FAIL",
        "model_version": manifest["model_version"], "excluded_images_by_reason": excluded,
        "train_images": len(y_train), "val_images": len(y_val),
        "train_scenes": len({row["scene_id"] for row in rows["train"]}),
        "val_scenes": len({row["scene_id"] for row in rows["val"]}),
        "support_PASS": int((y_val == 0).sum()), "support_FAIL": int((y_val == 1).sum()),
        "tn": int(tn), "false_FAIL": int(fp), "false_PASS": int(fn), "tp": int(tp),
        "FAIL_precision": float(p), "FAIL_recall": float(r), "FAIL_f1": float(f1),
        "misclassified": [
            {"image_id": row["image_id"], "scene_id": row["scene_id"],
             "true_label": ("PASS", "FAIL")[truth], "predicted_label": ("PASS", "FAIL")[pred],
             "p_fail": float(score)}
            for row, truth, pred, score in zip(rows["val"], y_val, predicted, scores)
            if truth != pred
        ],
    }
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".baseline2-", dir=args.output_dir.parent) as temp:
        bundle = Path(temp) / "bundle"
        bundle.mkdir()
        shutil.copyfile(checkpoint, bundle / WEIGHT_FILE)
        write_json(bundle / "baseline2.json", manifest)
        write_json(bundle / "development_inputs.json", input_rows)
        write_json(bundle / "validation_metrics.json", validation)
        with (bundle / "validation_images.csv").open("w", newline="", encoding="utf-8") as target:
            writer = csv.writer(target)
            writer.writerow(IMAGE_FIELDS)
            for row, pred, score in zip(rows["val"], predicted, scores):
                label = ("PASS", "FAIL")[pred]
                writer.writerow((row["image_id"], MODEL_ID, label, label, float(score), "prob", ""))
        bundle.rename(args.output_dir.resolve())
    print(f"Saved {args.output_dir} (model_version={manifest['model_version']})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--images-csv", required=True, type=Path)
    parser.add_argument("--objects-csv", required=True, type=Path)
    parser.add_argument("--expected-counts", required=True, nargs=5, type=int, metavar="COUNT")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--weights", type=Path, help="Official local IMAGENET1K_V1 checkpoint")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
