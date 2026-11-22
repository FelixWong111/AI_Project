"""Ground-truth loading and sample building for baseline1.

The ground-truth image manifest follows the project contract
(``ground_truth_images.csv``)::

    image_id, scene_id, split, width, height, annotation_status,
    count_1, count_2, count_3, count_4, count_5, non_catalogue_count, label

``label`` is the derived image-level PASS/FAIL label under the evaluated
expected-count vector. For the image-level baseline1 it is the training
target. Object-level boxes live in a separate ``ground_truth_objects.csv``
and are NOT read by this image-only baseline.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import pandas as pd

from .catalogue import IMAGE_LABELS

COUNT_COLUMNS = [f"count_{i}" for i in range(1, 6)]
GT_COLUMNS = (
    ["image_id", "scene_id", "split", "width", "height", "annotation_status"]
    + COUNT_COLUMNS
    + ["non_catalogue_count", "label"]
)


def load_ground_truth(csv_path: str) -> pd.DataFrame:
    """Load and validate the ground-truth image manifest."""
    df = pd.read_csv(csv_path)
    missing = [c for c in GT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Ground-truth CSV missing columns: {missing}")
    df["label"] = df["label"].astype(str).str.strip().str.upper()
    bad = df[~df["label"].isin(IMAGE_LABELS)]
    if not bad.empty:
        raise ValueError(f"Invalid image labels found: {bad['label'].unique().tolist()}")
    for col in COUNT_COLUMNS + ["non_catalogue_count"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    return df.reset_index(drop=True)


def _resolve_in_dir(directory: str, image_id: str) -> Optional[str]:
    direct = os.path.join(directory, image_id)
    if os.path.isfile(direct):
        return direct
    base, ext = os.path.splitext(image_id)
    if ext:
        return None
    for suffix in (".jpg", ".jpeg", ".png", ".bmp"):
        candidate = os.path.join(directory, base + suffix)
        if os.path.isfile(candidate):
            return candidate
    return None


def find_image_path(images_dir: str, image_id: str,
                    split: Optional[str] = None) -> Optional[str]:
    """Resolve an image_id under ``images_dir`` (and optional ``split`` subdir).

    Accepts an image_id with or without an extension.
    """
    resolved = _resolve_in_dir(images_dir, image_id)
    if resolved is not None:
        return resolved
    if split:
        resolved = _resolve_in_dir(os.path.join(images_dir, split), image_id)
        if resolved is not None:
            return resolved
    return None


def build_samples(gt_df: pd.DataFrame, images_dir: str,
                  split: Optional[str] = None) -> List[Dict]:
    """Build sample records ``{image_id, label, path}`` for a split.

    Missing images are skipped and reported; callers decide whether that
    is fatal for their run.
    """
    if split is not None:
        gt_df = gt_df[gt_df["split"] == split]
    samples: List[Dict] = []
    missing: List[str] = []
    for _, row in gt_df.iterrows():
        image_id = str(row["image_id"])
        row_split = str(row["split"]) if "split" in row.index else split
        path = find_image_path(images_dir, image_id, row_split)
        if path is None:
            missing.append(image_id)
            continue
        samples.append({
            "image_id": image_id,
            "label": row["label"],
            "path": path,
        })
    if missing:
        print(f"[dataset] WARNING: {len(missing)} image(s) not found, skipped: "
              f"{missing[:5]}{' ...' if len(missing) > 5 else ''}")
    return samples
