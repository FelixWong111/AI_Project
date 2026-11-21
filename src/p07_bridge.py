"""P07-local proposal, NOT the P08 public API. See README_P07.md."""
from __future__ import annotations

import importlib
import math
from typing import Any, Mapping

import numpy as np

MODELS = ("baseline1", "baseline2", "baseline3", "final")
CLASSES = ("non_catalogue", "long_screw", "thick_screw", "thin_screw", "hex_nut", "flat_washer")
STATUSES = {"PASS", "FAIL", "REVIEW", "UNSUPPORTED_COUNTS", "ERROR"}
SCENARIOS = ("pass", "missing", "extra", "id0", "empty", "only_id0", "review", "error")


def validate_counts(value: Mapping[str, int]) -> dict[str, int]:
    if not isinstance(value, dict) or set(value) != set("12345"):
        raise ValueError('expected_counts must contain exactly string keys "1".."5"')
    if any(type(v) is not int or v < 0 for v in value.values()):
        raise ValueError("counts must be non-negative integers (not bools)")
    return dict(value)


def validate_frame(frame: np.ndarray) -> None:
    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) < 1:
        raise ValueError("frame must be nonempty uint8 HxWx3 BGR")


def validate_result(result: dict, model: str, shape: tuple) -> dict:
    """Reject malformed bridge responses rather than invent missing object fields."""
    if not isinstance(result, dict) or result.get("status") not in STATUSES:
        raise ValueError("bridge returned invalid status")
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


class MockBackend:
    """Synthetic fixtures only. Observations never depend on expected_counts."""
    is_mock = True

    def __init__(self, model: str, config: dict):
        self.model = model
        self.fixed = validate_counts(config["mock_fixed_counts"])
        self.metadata = {"model_version": "MOCK-fixture-v1", "model_file_size_bytes": None,
                         "score_convention": "MOCK synthetic score; not calibrated probability"}

    def synchronize(self) -> None:
        pass

    def infer(self, frame: np.ndarray, expected: dict, scenario: str = "pass") -> dict:
        validate_frame(frame)
        validate_counts(expected)
        if scenario not in SCENARIOS:
            raise ValueError("unknown MOCK scenario")
        if scenario == "error":
            raise ValueError("MOCK injected inference error")
        if self.model in MODELS[:2]:
            self.image_result = dict(status="REVIEW" if scenario == "review" else ("PASS" if scenario == "pass" else "FAIL"),
                                     predicted_label=None, prediction_score=None, score_type=None,
                                     non_catalogue_count=None, counts=None, objects=[])
            if self.image_result["status"] in ("PASS", "FAIL"):
                self.image_result.update(predicted_label=self.image_result["status"], prediction_score=0.9, score_type="prob")
            return self.rejudge(self.image_result, expected)
        observed = {str(i): 1 for i in range(1, 6)}
        observed["0"] = 0
        if scenario in ("empty", "only_id0"):
            observed = dict.fromkeys("012345", 0)
        if scenario == "missing":
            observed["1"] = 0
        if scenario == "extra":
            observed["2"] = 2
        if scenario in ("id0", "only_id0"):
            observed["0"] = 1
        h, w = frame.shape[:2]
        objects = []
        for cid in range(6):
            for j in range(observed[str(cid)]):
                n = len(objects)
                x, y = (n % 4) * w / 4, (n // 4) * h / 3
                objects.append(dict(detection_id=n, class_id=cid, xmin=x, ymin=y,
                                    xmax=x + w / 6, ymax=y + h / 5, score=0.9))
        raw = dict(status="REVIEW" if scenario == "review" else "PASS", predicted_label=None,
                   prediction_score=None, score_type=None, non_catalogue_count=observed["0"],
                   counts=observed, objects=objects)
        return self.rejudge(raw, expected)

    def rejudge(self, result: dict, expected: dict) -> dict:
        validate_counts(expected)
        new = dict(result)
        if self.model in MODELS[:2]:
            if expected != self.fixed:
                new.update(status="UNSUPPORTED_COUNTS", predicted_label=None, prediction_score=None, score_type=None)
                return new
            return dict(getattr(self, "image_result", new))
        if new["status"] not in ("REVIEW", "ERROR"):
            status = "PASS" if new["counts"]["0"] == 0 and all(new["counts"][c] == expected[c] for c in "12345") else "FAIL"
            new.update(status=status, predicted_label=status)
        return new


def load_backend(mode: str, model: str, config: dict) -> Any:
    if model not in MODELS:
        raise ValueError("unknown model ID")
    if mode == "mock":
        return MockBackend(model, config)
    target = config.get("bridge")
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
    return backend
