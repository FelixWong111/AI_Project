"""Generate a tiny SYNTHETIC demo dataset to smoke-test the baseline1 pipeline.

WARNING
-------
These images are procedurally generated placeholders, NOT real inspection
photos. They exist only so the train -> freeze -> predict pipeline can be
run end-to-end before P02/P03 deliver the real photographed and annotated
dataset. Do not report demo metrics as project evidence.

Usage
-----
    python tools/make_demo_data.py --out data/demo
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.p04.catalogue import TARGET_CLASSES  # noqa: E402

IMG_SIZE = 128
SEED = 42
SUPPORTED_COUNTS = {"1": 1, "2": 1, "3": 1, "4": 1, "5": 1}

SPLITS = {"train": 40, "val": 16, "test": 16}


def _draw_pass(rng: np.random.Generator, size: int) -> Image.Image:
    """PASS scene: light-gray small parts on a clean white background."""
    img = Image.new("RGB", (size, size), (245, 245, 245))
    draw = ImageDraw.Draw(img)
    n_parts = int(rng.integers(3, 6))
    for _ in range(n_parts):
        x = int(rng.integers(8, size - 28))
        y = int(rng.integers(8, size - 28))
        w = int(rng.integers(10, 22))
        h = int(rng.integers(6, 12))
        shade = int(rng.integers(150, 190))
        draw.rounded_rectangle([x, y, x + w, y + h], radius=3,
                               fill=(shade, shade, shade))
    arr = np.asarray(img).astype(np.int16)
    arr += rng.integers(-6, 7, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _draw_fail(rng: np.random.Generator, size: int) -> Image.Image:
    """FAIL scene: a dark non-catalogue blob on a white background."""
    img = Image.new("RGB", (size, size), (245, 245, 245))
    draw = ImageDraw.Draw(img)
    x = int(rng.integers(20, size - 40))
    y = int(rng.integers(20, size - 40))
    r = int(rng.integers(14, 22))
    draw.ellipse([x, y, x + 2 * r, y + 2 * r], fill=(35, 45, 90))
    # a couple of light parts as well, so the scene is not all dark
    for _ in range(2):
        px = int(rng.integers(8, size - 24))
        py = int(rng.integers(8, size - 16))
        draw.rounded_rectangle([px, py, px + 14, py + 8], radius=3,
                               fill=(170, 170, 170))
    arr = np.asarray(img).astype(np.int16)
    arr += rng.integers(-6, 7, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join(REPO_ROOT, "data", "demo"))
    args = parser.parse_args()

    rng = np.random.default_rng(SEED)
    rows = []
    expected_cases = []
    image_id = 0

    for split, n_total in SPLITS.items():
        split_dir = os.path.join(args.out, "images", split)
        os.makedirs(split_dir, exist_ok=True)
        n_fail = n_total // 2
        for i in range(n_total):
            is_fail = i < n_fail
            label = "FAIL" if is_fail else "PASS"
            img = _draw_fail(rng, IMG_SIZE) if is_fail else _draw_pass(rng, IMG_SIZE)
            image_id += 1
            fname = f"{split}_{image_id:04d}.png"
            img.save(os.path.join(split_dir, fname))

            counts = dict(SUPPORTED_COUNTS)
            non_cat = 1 if is_fail else 0
            if is_fail and rng.random() < 0.5:
                # alternate failure mode: a target count is off, no non-catalogue
                non_cat = 0
                off_class = str(int(rng.integers(1, 6)))
                counts[off_class] = max(0, counts[off_class] - 1)

            row = {
                "image_id": fname,
                "scene_id": f"{split}_scene_{i:03d}",
                "split": split,
                "width": IMG_SIZE,
                "height": IMG_SIZE,
                "annotation_status": "ok",
                "count_1": counts["1"],
                "count_2": counts["2"],
                "count_3": counts["3"],
                "count_4": counts["4"],
                "count_5": counts["5"],
                "non_catalogue_count": non_cat,
                "label": label,
            }
            rows.append(row)
            if split == "test":
                expected_cases.append({
                    "image_id": fname,
                    "expected_counts": SUPPORTED_COUNTS,
                })

    import csv
    csv_path = os.path.join(args.out, "ground_truth_images.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with open(os.path.join(args.out, "expected_counts.json"), "w", encoding="utf-8") as f:
        json.dump(expected_cases, f, indent=2)

    print(f"[demo] wrote {len(rows)} images + manifest to {args.out}")
    print(f"[demo] target catalogue: {[c['class_name'] for c in TARGET_CLASSES]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
