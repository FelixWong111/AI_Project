"""Frozen ResNet18 image-level PASS/FAIL adapter for SW-02."""

from __future__ import annotations

import hashlib
import json
import math
import platform
from pathlib import Path
from typing import Mapping

import numpy as np
from PIL import Image, ImageOps

MODEL_ID = "baseline2"
IMAGE_FIELDS = (
    "image_id", "model_id", "status", "predicted_label", "prediction_score",
    "score_type", "non_catalogue_count",
)
CLASS_CATALOGUE = [
    {"id": 0, "name": "non_catalogue", "spec": "type/spec not in catalogue"},
    {"id": 1, "name": "long_screw", "spec": "M4x25"},
    {"id": 2, "name": "thick_screw", "spec": "M5x10"},
    {"id": 3, "name": "thin_screw", "spec": "M4x10"},
    {"id": 4, "name": "hex_nut", "spec": "M4"},
    {"id": 5, "name": "flat_washer", "spec": "M5"},
]
COUNT_KEYS = tuple(str(n) for n in range(1, 6))
WEIGHT_FILE = "resnet18-f37072fd.pth"
WEIGHT_SHA256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
PREPROCESSING = {
    "color": "RGB", "orientation": "EXIF for file inputs",
    "resize": "aspect-preserving fit into 224x224, white padding, bilinear",
    "normalization_mean": [0.485, 0.456, 0.406],
    "normalization_std": [0.229, 0.224, 0.225],
    "feature_layer": "global average pool, 512 dimensions",
}


def validate_counts(counts: Mapping[str, int]) -> dict[str, int]:
    if not isinstance(counts, Mapping) or set(counts) != set(COUNT_KEYS):
        raise ValueError("expected_counts must contain exactly string keys 1–5")
    if any(type(value) is not int or value < 0 for value in counts.values()):
        raise ValueError("expected_counts values must be non-negative integers")
    return {key: counts[key] for key in COUNT_KEYS}


def image_filename(value: str) -> str:
    if not isinstance(value, str) or not value or value in {".", ".."}:
        raise ValueError("image_id must be a filename")
    if any(token in value for token in ("/", "\\", ":", "\x00")):
        raise ValueError(f"Invalid image_id: {value!r}")
    return value


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def model_version(manifest: dict) -> str:
    payload = {key: value for key, value in manifest.items() if key != "model_version"}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def read_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


class ResNet18Features:
    """Load a local official ImageNet checkpoint and extract one vector per image."""

    def __init__(self, weight_path: Path, device: str = "cpu"):
        import torch
        from torchvision import transforms
        from torchvision.models import resnet18

        self.torch = torch
        self.device = torch.device(device)
        self.backbone = resnet18(weights=None)
        self.backbone.load_state_dict(
            torch.load(weight_path, map_location="cpu", weights_only=True)
        )
        self.backbone.fc = torch.nn.Identity()
        self.backbone.requires_grad_(False)
        self.backbone.eval().to(self.device)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                PREPROCESSING["normalization_mean"], PREPROCESSING["normalization_std"]
            ),
        ])

    def extract(self, images: list[Image.Image]) -> np.ndarray:
        tensors = [
            self.transform(ImageOps.pad(
                image, (224, 224), Image.Resampling.BILINEAR, (255, 255, 255)
            ))
            for image in images
        ]
        batch = self.torch.stack(tensors).to(self.device)
        with self.torch.inference_mode():
            return self.backbone(batch).cpu().numpy().astype(np.float64)


class Baseline2:
    """Shared entry point for offline image files and OpenCV BGR frames."""

    def __init__(self, bundle_dir: str | Path, device: str = "cpu"):
        bundle = Path(bundle_dir)
        self.manifest = json.loads((bundle / "baseline2.json").read_text(encoding="utf-8"))
        if (self.manifest.get("schema") != "baseline2/v1"
                or self.manifest.get("model_version") != model_version(self.manifest)
                or self.manifest.get("preprocessing") != PREPROCESSING):
            raise ValueError("baseline2 model version, schema or preprocessing mismatch")
        self.fixed_counts = validate_counts(self.manifest["supported_fixed_counts"])
        params = self.manifest["classifier"]
        self.mean, self.scale, self.coef = (
            np.asarray(params[key], dtype=np.float64) for key in ("mean", "scale", "coef")
        )
        self.intercept = float(params["intercept"])
        self.threshold = float(params["fail_probability_threshold"])
        if (any(array.shape != (512,) or not np.isfinite(array).all()
                for array in (self.mean, self.scale, self.coef))
                or np.any(self.scale <= 0)
                or not math.isfinite(self.intercept)
                or not 0 < self.threshold < 1):
            raise ValueError("Invalid baseline2 classifier parameters")
        checkpoint = bundle / WEIGHT_FILE
        if not sha256(checkpoint) == self.manifest["backbone_sha256"] == WEIGHT_SHA256:
            raise ValueError("ResNet18 checkpoint SHA-256 mismatch")
        self.features = ResNet18Features(checkpoint, device)
        self.device = str(self.features.device)

    def _result(self, image_id: str, status: str, score: float | None = None) -> dict:
        return dict(zip(IMAGE_FIELDS, (
            image_id, MODEL_ID, status, status if score is not None else None,
            score, "prob" if score is not None else "none", None,
        )))

    def _counts_status(self, expected_counts: Mapping[str, int]) -> str | None:
        try:
            counts = validate_counts(expected_counts)
        except ValueError:
            return "ERROR"
        return None if counts == self.fixed_counts else "UNSUPPORTED_COUNTS"

    def _predict_rgb(self, image_id: str, image: Image.Image) -> dict:
        features = self.features.extract([image])[0]
        logit = float(((features - self.mean) / self.scale) @ self.coef + self.intercept)
        if not math.isfinite(logit):
            raise ValueError("Non-finite baseline2 score")
        exp_negative_magnitude = math.exp(-abs(logit))
        denominator = 1 + exp_negative_magnitude
        score = 1 / denominator if logit >= 0 else exp_negative_magnitude / denominator
        return self._result(image_id, "FAIL" if score >= self.threshold else "PASS", score)

    def predict_path(self, path: str | Path, expected_counts: Mapping[str, int],
                     *, image_id: str | None = None) -> dict:
        path = Path(path)
        image_id = image_filename(image_id if image_id is not None else path.name)
        status = self._counts_status(expected_counts)
        if status is not None:
            return self._result(image_id, status)
        try:
            image = read_rgb(path)
        except (OSError, ValueError, Image.DecompressionBombError):
            return self._result(image_id, "ERROR")
        return self._predict_rgb(image_id, image)

    def predict_frame(self, frame_bgr: np.ndarray, expected_counts: Mapping[str, int],
                      *, image_id: str) -> dict:
        image_id = image_filename(image_id)
        status = self._counts_status(expected_counts)
        if status is not None:
            return self._result(image_id, status)
        if (not isinstance(frame_bgr, np.ndarray) or frame_bgr.dtype != np.uint8
                or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3 or not frame_bgr.size):
            return self._result(image_id, "ERROR")
        rgb = Image.fromarray(np.ascontiguousarray(frame_bgr[..., ::-1]))
        return self._predict_rgb(image_id, rgb)

    def run_config(self, per_image_expected_counts: dict) -> dict:
        device_name = platform.processor() or platform.machine()
        if self.features.device.type == "cuda":
            device_name = self.features.torch.cuda.get_device_name(self.features.device)
        return {
            "model": MODEL_ID,
            "model_version": self.manifest["model_version"],
            "output_capability": "image_level",
            "class_catalogue": CLASS_CATALOGUE,
            "thresholds": {"fail_probability": self.threshold, "rule": ">="},
            "preprocessing": PREPROCESSING,
            "coordinate_convention": None,
            "score_convention": {"prob": "estimated P(FAIL), no probability calibration"},
            "hardware": {"device": self.device, "device_name": device_name},
            "rejection": None,
            "size_tolerance": None,
            "calibration": None,
            "per_image_expected_counts": per_image_expected_counts,
            "supported_fixed_counts": self.fixed_counts,
        }
