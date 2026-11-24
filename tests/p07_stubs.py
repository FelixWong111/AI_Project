"""STUB_PASS fixtures, importable only from tests; not deployment models."""
from src.p07_bridge import ImageBackend, MODELS, SCENARIOS, validate_frame, validate_counts
import numpy as np
from src.work_orders import reference_counts


class MockBackend:
    """Synthetic fixtures only. Observations never depend on expected_counts."""
    is_mock = True

    def __init__(self, model: str, config: dict):
        if model not in MODELS[2:]:
            raise ValueError("MOCK is only available for baseline3/final; baseline1/2 require real artifacts")
        self.model = model
        self.fixed = validate_counts(config.get("mock_fixed_counts", config.get("expected_counts", reference_counts())))
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
        if new["status"] not in ("REVIEW", "ERROR"):
            status = "PASS" if new["counts"]["0"] == 0 and all(new["counts"][c] == expected[c] for c in "12345") else "FAIL"
            new.update(status=status, predicted_label=status)
        return new


class ImageStub:
    def __init__(self, model, fixed):
        self.model, self.fixed, self.calls = model, dict(fixed), 0
        self.metadata = dict(model_version="TEST-STUB-ONLY", supported_fixed_counts=self.fixed,
                             output_capability="image_level", model_file_size_bytes=None, evidence_scope="TEST_STUB_ONLY")

    def synchronize(self):
        pass

    def infer(self, frame, expected):
        self.calls += 1
        return dict(status="PASS", predicted_label="PASS", prediction_score=.8, score_type="prob",
                    counts=None, objects=[], non_catalogue_count=None)


def fixture_backend(model, config):
    return ImageBackend(model, ImageStub(model, config.get("expected_counts", reference_counts())), []) if model in ("baseline1", "baseline2") else MockBackend(model, config)
