"""Standalone baseline2 batch runner for P05 handoff and integration checks."""

import argparse
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from src.adapters.baseline2 import Baseline2, IMAGE_FIELDS, MODEL_ID, image_filename


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument("--model", required=True, choices=[MODEL_ID])
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--counts_file", required=True, type=Path)
    parser.add_argument("--bundle-dir", default=Path("weights/baseline2"), type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if not args.input_dir.is_dir():
        parser.error("--input_dir must be an existing directory")
    if args.output_dir.exists():
        parser.error("--output_dir must be a new directory")
    cases = json.loads(args.counts_file.read_text(encoding="utf-8-sig"))
    if not isinstance(cases, list) or not cases:
        parser.error("--counts_file must contain a non-empty JSON array")
    per_image = {}
    for case in cases:
        if not isinstance(case, dict) or set(case) != {"image_id", "expected_counts"}:
            parser.error("Each case must contain image_id and expected_counts")
        image_id = image_filename(case["image_id"])
        if image_id in per_image:
            parser.error(f"Duplicate image_id: {image_id}")
        per_image[image_id] = case["expected_counts"]
    adapter = Baseline2(args.bundle_dir, args.device)
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".p05-prediction-", dir=args.output_dir.parent) as temp:
        staging = Path(temp) / "output"
        staging.mkdir()
        with (staging / "images.csv").open("w", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, fieldnames=IMAGE_FIELDS)
            writer.writeheader()
            for image_id, counts in per_image.items():
                writer.writerow(adapter.predict_path(args.input_dir / image_id, counts))
        (staging / "run_config.json").write_text(
            json.dumps(adapter.run_config(per_image), ensure_ascii=False, indent=2,
                       allow_nan=False) + "\n",
            encoding="utf-8",
        )
        staging.rename(args.output_dir.resolve())


if __name__ == "__main__":
    main()
