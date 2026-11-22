"""Exercise the P05 pipeline with generated images and official ImageNet weights."""

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw

from src.adapters.baseline2 import Baseline2, COUNT_KEYS, IMAGE_FIELDS, read_rgb, sha256
from src.stage3_clahe import DetectorCLAHE


def write_csv(path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.writer(target)
        writer.writerow(header)
        writer.writerows(rows)


def make_images(root):
    image_dir = root / "images"
    image_dir.mkdir()
    images, objects = [], []
    rng = np.random.default_rng(42)
    for number in range(20):
        image_id = f"synthetic_{number:02d}.png"
        split = "train" if number < 12 else "val"
        images.append((image_id, f"synthetic_scene_{number:02d}", split, "OK"))
        modes = ([1, 2, 3, 4, 5], [1, 2, 3, 4],
                 [1, 2, 3, 4, 5, 0], [1, 2, 3, 4, 5, 1], [0], [])
        classes = modes[0] if number % 2 == 0 else modes[1 + (number // 2) % 5]
        image = Image.new("RGB", (256, 192), (248, 248, 248))
        draw = ImageDraw.Draw(image)
        for obj, cls in enumerate(classes):
            x = 22 + obj % 3 * 77 + int(rng.integers(-5, 6))
            y = 29 + obj // 3 * 80 + int(rng.integers(-5, 6))
            w, h = (23 + 4 * cls, 13 + 3 * cls) if cls else (51, 48)
            box = (x, y, x + w, y + h)
            if cls in (4, 5):
                draw.ellipse(box, fill=(94, 97, 107), outline=(42, 42, 42), width=2)
            else:
                draw.rectangle(box, fill=(72 + 15 * cls, 81, 96), outline=(43, 43, 43))
            objects.append((image_id, obj, cls, *box, 0))
        if number == 0:
            objects.append((image_id, len(classes), 0, 0, 0, 8, 8, 1))
        image.save(image_dir / image_id)
    images.extend([
        ("review_not_loaded.png", "synthetic_review", "val", "REVIEW"),
        ("test_not_loaded.png", "synthetic_sealed", "test", "OK"),
    ])
    (image_dir / "broken.png").write_bytes(b"not an image")
    write_csv(root / "ground_truth_images.csv",
              ["image_id", "scene_id", "split", "annotation_status"], images)
    write_csv(root / "ground_truth_objects.csv",
              ["image_id", "object_id", "class_id", "xmin", "ymin", "xmax", "ymax", "ignore"],
              objects)
    return image_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=False)
    root = args.work_dir
    images = make_images(root)
    bundle = root / "bundle"
    fixed = dict.fromkeys(COUNT_KEYS, 1)
    alternate = fixed | {"5": 0}
    invalid = fixed | {"5": True}
    val_ids = [f"synthetic_{i:02d}.png" for i in range(12, 20)]
    cases = [(image_id, fixed) for image_id in val_ids] + [
        ("synthetic_00.png", alternate), ("synthetic_01.png", invalid),
        ("broken.png", fixed), ("missing.png", fixed),
    ]
    counts_path = root / "counts.json"
    counts_path.write_text(json.dumps([
        {"image_id": image_id, "expected_counts": counts} for image_id, counts in cases
    ]), encoding="utf-8")
    environment = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    command = [sys.executable, "-m", "scripts.train_baseline2",
               "--image-dir", str(images), "--images-csv", str(root / "ground_truth_images.csv"),
               "--objects-csv", str(root / "ground_truth_objects.csv"),
               "--expected-counts", "1", "1", "1", "1", "1", "--output-dir", str(bundle),
               "--weights", str(args.weights)]
    subprocess.run(command, check=True, env=environment)
    output = root / "predictions"
    subprocess.run(
        [sys.executable, "-m", "scripts.p05_predict", "--input_dir", str(images),
         "--model", "baseline2", "--output_dir", str(output), "--counts_file", str(counts_path),
         "--bundle-dir", str(bundle)], check=True, env=environment,
    )
    with (output / "images.csv").open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        assert reader.fieldnames == list(IMAGE_FIELDS)
        rows = {row["image_id"]: row for row in reader}
    assert len(rows) == len(cases)
    assert rows["synthetic_00.png"]["status"] == "UNSUPPORTED_COUNTS"
    assert not rows["synthetic_00.png"]["predicted_label"]
    assert not rows["synthetic_00.png"]["prediction_score"]
    assert all(rows[i]["status"] == "ERROR" for i in ("synthetic_01.png", "broken.png", "missing.png"))
    assert all(not row["non_catalogue_count"] for row in rows.values())
    assert {p.name for p in output.iterdir()} == {"images.csv", "run_config.json"}
    config = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
    assert config["supported_fixed_counts"] == fixed
    assert config["output_capability"] == "image_level"
    assert config["per_image_expected_counts"]["synthetic_00.png"] == alternate
    assert config["per_image_expected_counts"]["synthetic_12.png"] == fixed

    with (bundle / "validation_images.csv").open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            actual = rows[row["image_id"]]
            assert row["predicted_label"] == actual["predicted_label"]
            assert np.isclose(float(row["prediction_score"]), float(actual["prediction_score"]),
                              rtol=1e-5, atol=1e-6)

    import cv2
    import torch

    torch.set_num_threads(2)
    adapter = Baseline2(bundle)
    path = images / val_ids[0]
    bgr = np.asarray(read_rgb(path))[..., ::-1].copy()
    assert adapter.predict_path(path, fixed) == adapter.predict_frame(bgr, fixed, image_id=path.name)
    original = bgr.copy()
    enhanced = DetectorCLAHE(2.0, (8, 8))(bgr)
    assert enhanced.shape == bgr.shape and enhanced.dtype == bgr.dtype
    assert np.array_equal(original, bgr) and not np.array_equal(enhanced, bgr)

    # An image with a scene in two partitions must be rejected before feature extraction.
    from scripts.train_baseline2 import load_development_rows

    manifest = root / "ground_truth_images.csv"
    invalid_split = root / "overlap.csv"
    invalid_split.write_text(manifest.read_text(encoding="utf-8") +
                             "synthetic_overlap.png,synthetic_scene_00,test,OK\n", encoding="utf-8")
    from types import SimpleNamespace
    try:
        load_development_rows(
            SimpleNamespace(images_csv=invalid_split, objects_csv=root / "ground_truth_objects.csv",
                            image_dir=images), fixed,
        )
    except ValueError as error:
        assert "more than one split" in str(error)
    else:
        raise AssertionError("Scene leakage was accepted")

    manifest_json = json.loads((bundle / "baseline2.json").read_text(encoding="utf-8"))
    metrics = json.loads((bundle / "validation_metrics.json").read_text(encoding="utf-8"))
    assert metrics["excluded_images_by_reason"] == {"val:REVIEW": 1}
    evidence = {
        "scope": "synthetic interface verification, no course performance data",
        "status": "PASS",
        "checkpoint_sha256": sha256(args.weights),
        "synthetic_model_version": config["model_version"],
        "versions": manifest_json["versions"] | {"opencv": cv2.__version__},
        "train_scenes": 12, "validation_scenes": 8,
        "checks": [
            "official ResNet18 features and sklearn fit",
            "sealed test placeholder not opened; REVIEW excluded",
            "scene overlap rejected",
            "validation probabilities matched reloaded classifier",
            "offline path and BGR frame returned identical results",
            "fixed counts, unsupported counts, malformed counts, missing and broken images",
            "images.csv schema and run_config.json",
            "CLAHE preserved dimensions and did not mutate the source",
        ],
        "real_course_training": "NOT_RUN",
        "real_stage3_comparison": "NOT_RUN",
    }
    (root / "smoke_evidence.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
