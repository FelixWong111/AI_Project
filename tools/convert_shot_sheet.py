"""Convert the team's shot annotation sheet into the project ground-truth manifest.

The raw sheet (拍摄填写表.csv, GBK-encoded) uses Chinese working columns:
文件名, 场景号, 组, e1..e5, a1..a5, a0, 自动判定, 应为, 核对, ...

This tool maps it to the contract ``ground_truth_images.csv`` columns:
image_id, scene_id, split, width, height, annotation_status,
count_1..count_5, non_catalogue_count, label.

Rows marked 核对 != OK (e.g. 待定 / 待复核) are excluded from train/val and
must not be treated as labeled evidence.

Usage
-----
    python tools/convert_shot_sheet.py \
        --sheet data/raw/照片/测试/拍摄填写表.csv \
        --images-dir data/raw/照片/测试 \
        --out-csv data/ground_truth_images.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Dict, List

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.p04 import dataset, features  # noqa: E402

CONTRACT_COLUMNS = [
    "image_id", "scene_id", "split", "width", "height", "annotation_status",
    "count_1", "count_2", "count_3", "count_4", "count_5",
    "non_catalogue_count", "label",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert shot sheet to ground-truth manifest")
    p.add_argument("--sheet", required=True, help="raw annotation sheet CSV (GBK)")
    p.add_argument("--images-dir", required=True, help="folder containing the sheet images")
    p.add_argument("--out-csv", required=True, help="output contract manifest CSV")
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--encoding", default="gbk")
    return p.parse_args()


def _read_sheet(path: str, encoding: str) -> List[Dict]:
    import pandas as pd
    df = pd.read_csv(path, encoding=encoding)
    records = df.to_dict(orient="records")
    return records


def main() -> int:
    args = parse_args()
    rows = _read_sheet(args.sheet, args.encoding)

    kept: List[Dict] = []
    excluded: List[str] = []
    for row in rows:
        label = str(row.get("应为", "")).strip().upper()
        check = str(row.get("核对", "")).strip()
        image_id = str(row.get("文件名", "")).strip()
        if label not in ("PASS", "FAIL") or check != "OK":
            excluded.append(image_id)
            continue
        kept.append(row)

    # Deterministic stratified split (each scene is a single frame here).
    rng = __import__("random").Random(args.seed)
    by_label: Dict[str, List[Dict]] = {"PASS": [], "FAIL": []}
    for row in kept:
        by_label[str(row["应为"]).strip().upper()].append(row)

    train_rows: List[Dict] = []
    val_rows: List[Dict] = []
    for label, group in by_label.items():
        rng.shuffle(group)
        n_val = max(1, round(len(group) * args.val_fraction)) if group else 0
        val_rows.extend(group[:n_val])
        train_rows.extend(group[n_val:])

    def _manifest_row(row: Dict, split: str) -> Dict:
        image_id = str(row["文件名"]).strip()
        scene_id = str(row["场景号"]).strip()
        width = height = ""
        img_path = dataset.find_image_path(args.images_dir, image_id)
        if img_path is not None and os.path.isfile(img_path):
            bgr = features.load_image_bgr(img_path)
            height, width = bgr.shape[:2]
        return {
            "image_id": image_id,
            "scene_id": scene_id,
            "split": split,
            "width": width,
            "height": height,
            "annotation_status": str(row.get("核对", "")).strip(),
            "count_1": int(row["a1"]),
            "count_2": int(row["a2"]),
            "count_3": int(row["a3"]),
            "count_4": int(row["a4"]),
            "count_5": int(row["a5"]),
            "non_catalogue_count": int(row["a0"]),
            "label": str(row["应为"]).strip().upper(),
        }

    manifest = [_manifest_row(r, "train") for r in train_rows]
    manifest += [_manifest_row(r, "val") for r in val_rows]

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CONTRACT_COLUMNS)
        writer.writeheader()
        writer.writerows(manifest)

    n_pass = sum(1 for r in manifest if r["label"] == "PASS")
    n_fail = sum(1 for r in manifest if r["label"] == "FAIL")
    print(f"[convert] kept={len(manifest)} (PASS={n_pass}, FAIL={n_fail}) "
          f"train={len(train_rows)} val={len(val_rows)}")
    if excluded:
        print(f"[convert] excluded {len(excluded)} unconfirmed rows: {excluded}")
    print(f"[convert] manifest -> {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
