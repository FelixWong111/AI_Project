"""P07-local proposal, NOT the P08 public API. See README_P07.md."""
from __future__ import annotations

import importlib
import math
import json
from pathlib import Path
from typing import Any

import numpy as np
from src.work_orders import REFERENCE_CONFIG, validate_counts, reference_counts, unsupported_counts, judge_observed

MODELS = ("baseline1", "baseline2", "baseline3", "final")
CLASSES = ("non_catalogue", "long_screw", "thick_screw", "thin_screw", "hex_nut", "flat_washer")
STATUSES = {"PASS", "FAIL", "REVIEW", "UNSUPPORTED_COUNTS", "ERROR"}
SCENARIOS = ("pass", "missing", "extra", "id0", "empty", "only_id0", "review", "error")
REPO_ROOT = Path(__file__).resolve().parents[1]


def validate_frame(frame: np.ndarray) -> None:
    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) < 1:
        raise ValueError("frame must be nonempty uint8 HxWx3 BGR")


def validate_result(result: dict, model: str, shape: tuple) -> dict:
    """Reject malformed bridge responses rather than invent missing object fields."""
    if not isinstance(result, dict) or result.get("status") not in STATUSES:
        raise ValueError("bridge returned invalid status")
    if result.get("model_id", model) != model:
        raise ValueError("result model_id does not match the selected model")
    if result.get("predicted_label") not in (None, "PASS", "FAIL"):
        raise ValueError("invalid predicted_label")
    if result.get("status") in {"REVIEW", "ERROR", "UNSUPPORTED_COUNTS"} and result.get("predicted_label") is not None:
        raise ValueError("non-success result must have null label")
    if result["status"] in {"PASS", "FAIL"} and result.get("predicted_label") != result["status"]:
        raise ValueError("decision and status disagree")
    if result.get("score_type") not in (None, "prob", "logit", "none"):
        raise ValueError("invalid score_type")
    score = result.get("prediction_score")
    if score is not None and (type(score) not in (int, float) or not math.isfinite(score)):
        raise ValueError("invalid prediction_score")
    if score is not None and result.get("score_type") == "prob" and not 0 <= score <= 1:
        raise ValueError("probability outside [0,1]")
    if result["status"] == "UNSUPPORTED_COUNTS" and score is not None:
        raise ValueError("unsupported counts must have null score")
    if model in MODELS[:2]:
        if result.get("objects") or result.get("counts") is not None or result.get("non_catalogue_count") is not None:
            raise ValueError("image-level backend fabricated object outputs")
    elif result["status"] != "ERROR":
        counts = result.get("counts")
        if result["status"] == "REVIEW" and counts is None and not result.get("objects"):
            return result
        if not isinstance(counts, dict) or set(counts) != set("012345") or any(type(v) is not int or v < 0 for v in counts.values()):
            raise ValueError("object-level counts must cover 0..5")
        objects = result.get("objects")
        if not isinstance(objects, list):
            raise ValueError("object list missing")
        observed = dict.fromkeys("012345", 0)
        ids = set()
        h, w = shape[:2]
        for obj in objects:
            cid = obj.get("class_id")
            if type(cid) is not int or cid not in range(6) or obj.get("detection_id") in ids:
                raise ValueError("invalid class or duplicate detection_id")
            ids.add(obj["detection_id"])
            coords = [obj[k] for k in ("xmin", "ymin", "xmax", "ymax")]
            if any(type(v) not in (int, float) or not math.isfinite(v) for v in coords):
                raise ValueError("nonfinite box")
            x1, y1, x2, y2 = coords
            if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
                raise ValueError("box outside original image")
            if not isinstance(obj.get("score"), (int, float)) or not math.isfinite(obj["score"]):
                raise ValueError("invalid object score")
            observed[str(cid)] += 1
        if counts != observed or result.get("non_catalogue_count") != counts["0"]:
            raise ValueError("objects and counts disagree")
        if counts["0"] and result.get("predicted_label") == "PASS":
            raise ValueError("ID 0 cannot PASS")
    return result



def repository_path(value):
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def image_spec(model, config):
    """Resolve owner-defined paths without moving or rewriting their files."""
    settings = dict(config.get("model_config", {}))
    if model == "baseline1":
        path = repository_path(settings.get("baseline1_config", "config/baseline1.json"))
        native = json.loads(path.read_text(encoding="utf-8-sig"))
        bundle = repository_path(settings.get("baseline1_bundle") or native["freeze"]["bundle_file"])
        files = [path, bundle, bundle.parent / "freeze.json", REFERENCE_CONFIG]
    else:
        path = repository_path("config/baseline2.json")
        defaults = json.loads(path.read_text(encoding="utf-8-sig"))
        settings = dict(defaults, **settings)
        bundle = repository_path(settings["baseline2_bundle"])
        settings["baseline2_bundle"] = str(bundle)
        files = [path, bundle / "baseline2.json", bundle / "resnet18-f37072fd.pth", REFERENCE_CONFIG]
    if config.get("_p07_config_path"):
        files.append(Path(config["_p07_config_path"]))
    return path, bundle, settings, files


def file_signature(files):
    return tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in files)


class ImageBackend:
    """P07 count/caching guard around a provider's real inference object."""
    is_mock = False

    def __init__(self, model, native, files):
        self.model, self.native = model, native
        self.fixed = validate_counts(native.fixed)
        self.metadata = dict(native.metadata)
        if self.metadata.get("model", model) != model:
            raise ValueError("loaded model metadata does not match the selected model")
        self.files, self.signature = files, file_signature(files)

    def synchronize(self):
        self.native.synchronize()

    def check_current(self):
        try:
            unchanged = file_signature(self.files) == self.signature
        except OSError:
            unchanged = False
        if not unchanged:
            raise RuntimeError("Model files/config changed or disappeared. Press L or click this model to reload")

    def _unsupported(self, cached=None, frame=None):
        result = dict(unsupported_counts(), _p07_prediction=cached,
                      _p07_needs_inference=cached is None, _p07_inference_executed=False)
        if frame is not None:
            result["_p07_pending_frame"] = frame.copy()
        return result

    def infer(self, frame_bgr, expected_counts):
        self.check_current()
        validate_frame(frame_bgr)
        expected = validate_counts(expected_counts)
        if expected != self.fixed:
            return self._unsupported(frame=frame_bgr)
        raw = self.native.infer(frame_bgr, expected)
        validate_result(raw, self.model, frame_bgr.shape)
        return dict(raw, _p07_prediction=dict(raw), _p07_inference_executed=True)

    def rejudge(self, result, expected_counts):
        self.check_current()
        expected = validate_counts(expected_counts)
        if result.get("status") == "ERROR":
            return dict(result)
        cached = result.get("_p07_prediction")
        if expected != self.fixed:
            return self._unsupported(cached, result.get("_p07_pending_frame"))
        if cached is None:
            # The frame was never inferred while counts were unsupported.
            return self.infer(result["_p07_pending_frame"], expected)
        return dict(cached, _p07_prediction=dict(cached), _p07_inference_executed=False)


class UnavailableBackend:
    is_mock = False

    def __init__(self, model, error):
        self.model, self.error = model, str(error)
        self.metadata = dict(model=model, model_version="UNAVAILABLE", model_file_size_bytes=None,
                             output_capability="image_level" if model in MODELS[:2] else "object_level",
                             error_message=self.error, deployment_stage="provisional" if model == "baseline1" else "submitted-final" if model == "baseline2" else "unavailable")

    def synchronize(self):
        pass

    def infer(self, frame_bgr=None, expected_counts=None):
        return dict(status="ERROR", predicted_label=None, prediction_score=None, score_type="none",
                    counts=None, objects=[], non_catalogue_count=None, message=self.error,
                    error_message=self.error, _p07_inference_executed=False)

    def rejudge(self, result, expected_counts):
        return self.infer()


class ObjectBackend:
    """Preserve detector observations when only the requested counts change."""
    is_mock = False

    def __init__(self, native, model):
        self.native, self.model = native, model
        self.metadata = dict(native.metadata)
        self.metadata['count_decision_score'] = 'none: decision derived from observed vs expected counts; object scores unchanged'

    def synchronize(self):
        self.native.synchronize()

    def infer(self, frame_bgr, expected_counts):
        expected = validate_counts(expected_counts)
        validate_frame(frame_bgr)
        raw = self.native.infer(frame_bgr, expected)
        result = judge_observed(raw, expected)
        validate_result(result, self.model, frame_bgr.shape)
        return result

    def rejudge(self, result, expected_counts):
        return judge_observed(result, expected_counts)


def load_backend(mode: str, model: str, config: dict) -> Any:
    if model not in MODELS:
        raise ValueError("unknown model ID")
    if model in MODELS[:2]:
        path, bundle, settings, files = image_spec(model, config)
        missing = [p.relative_to(REPO_ROOT).as_posix() if p.is_relative_to(REPO_ROOT) else str(p)
                   for p in files if not p.is_file()]
        if missing:
            raise FileNotFoundError(model + " missing artifacts: " + ", ".join(missing))
        before = file_signature(files)
        if model == "baseline1":
            from src.adapters.baseline1 import Baseline1Backend
            native = Baseline1Backend(path, REPO_ROOT, bundle)
            if config.get('_live_scene_guard'):
                from src.models.baseline1.scene_guard import SceneGuard
                guard_path = repository_path('models/baseline1/scene_guard.npz')
                native.scene_guard = SceneGuard(guard_path, native.metadata['model_version'])
                native.metadata['system_scene_guard'] = dict(enabled=True,
                    rule='nearest standardized HOG distance; threshold=max training leave-one-out distance',
                    threshold=native.scene_guard.limit, raw_baseline_unchanged=True,
                    scope='P07 system only; not raw baseline benchmark', artifact_sha256=__import__('hashlib').sha256(guard_path.read_bytes()).hexdigest())
                files.append(guard_path)
                before = before + file_signature([guard_path])
        else:
            # Reuse P05's loading and inference; do not duplicate ResNet/LR.
            from src.p05_bridge import create_backend
            native = create_backend(model_id=model, config=settings)
            native.metadata["deployment_stage"] = "candidate"
        if file_signature(files) != before:
            raise RuntimeError("Model files changed during loading; retry with a stable delivery set")
        return ImageBackend(model, native, files)
    target = config.get("bridge")
    if not target and model == 'baseline3':
        from src.adapters.baseline3 import create_backend
        return ObjectBackend(create_backend(model_id=model, config=config.get('model_config', {})), model)
    if not target or ":" not in target:
        raise RuntimeError("Real backend unavailable: configure P08 bridge as module:factory; no MOCK fallback")
    module, factory = target.split(":", 1)
    backend = getattr(importlib.import_module(module), factory)(model_id=model, config=config.get("model_config", {}))
    for method in ("infer", "rejudge", "synchronize"):
        if not callable(getattr(backend, method, None)):
            raise TypeError("P08 bridge missing " + method)
    if not isinstance(getattr(backend, "metadata", None), dict):
        raise TypeError("P08 bridge must supply metadata")
    backend.is_mock = False
    return ObjectBackend(backend, model)
