"""P05 baseline2 implementation of P07's proposed real-backend protocol."""

from __future__ import annotations

from pathlib import Path

from src.adapters.baseline2 import Baseline2, WEIGHT_FILE, validate_counts
from src.p07_bridge import validate_frame


class Baseline2Backend:
    is_mock = False

    def __init__(self, bundle_dir: str | Path, device: str = "cpu",
                 expected_model_version: str | None = None):
        self.adapter = Baseline2(bundle_dir, device)
        version = self.adapter.manifest["model_version"]
        if expected_model_version is not None and version != expected_model_version:
            raise ValueError("baseline2 model_version differs from the pinned configuration")
        self.fixed = dict(self.adapter.fixed_counts)
        self.metadata = self.adapter.run_config({})
        bundle = Path(bundle_dir)
        self.metadata["model_file_size_bytes"] = sum(
            (bundle / name).stat().st_size for name in ("baseline2.json", WEIGHT_FILE)
        )

    def synchronize(self) -> None:
        features = self.adapter.features
        if features.device.type == "cuda":
            features.torch.cuda.synchronize(features.device)

    def infer(self, frame_bgr, expected_counts: dict[str, int]) -> dict:
        validate_frame(frame_bgr)
        validate_counts(expected_counts)
        # Evaluate this frame under the trained rule once, even when the UI
        # currently requests another count vector. Rejudge can then restore it.
        prediction = self.adapter.predict_frame(
            frame_bgr, self.fixed, image_id="frame.png"
        )
        return self.rejudge({"_p05_prediction": prediction}, expected_counts)

    def rejudge(self, result: dict, expected_counts: dict[str, int]) -> dict:
        expected = validate_counts(expected_counts)
        original = result.get("_p05_prediction")
        if not isinstance(original, dict):
            # Preserve worker errors, which have no successful frame observation.
            if result.get("status") == "ERROR":
                return dict(result)
            raise ValueError("baseline2 cached prediction is missing")
        row = dict(original)
        if expected != self.fixed and row["status"] != "ERROR":
            row.update(status="UNSUPPORTED_COUNTS", predicted_label=None,
                       prediction_score=None, score_type="none")
        row.update(objects=[], counts=None, non_catalogue_count=None,
                   _p05_prediction=dict(original))
        return row


def create_backend(*, model_id: str, config: dict) -> Baseline2Backend:
    """Factory called by src.p07_bridge.load_backend in real mode."""
    if model_id != "baseline2":
        raise ValueError(f"P05 bridge supports baseline2, received {model_id!r}")
    if not isinstance(config, dict):
        raise ValueError("model_config must be an object")
    bundle = config.get("baseline2_bundle")
    if not isinstance(bundle, str) or not bundle:
        raise ValueError("model_config.baseline2_bundle must point to a trained bundle")
    return Baseline2Backend(
        bundle, device=config.get("device", "cpu"),
        expected_model_version=config.get("expected_model_version"),
    )
