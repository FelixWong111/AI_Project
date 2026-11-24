"""Result presenter and one background inference worker."""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

from .p07_bridge import CLASSES, validate_result
from .work_orders import reference_counts, unsupported_counts, validate_counts


def current_result(model, backend, expected, cached, pending=False):
    """Only present a decision for the currently selected model and requirement."""
    expected = validate_counts(expected)
    if model in ('baseline1', 'baseline2') and expected != reference_counts():
        return unsupported_counts()
    if pending:
        return None
    if cached and cached['model'] == model and cached['expected'] == expected:
        return cached['result']
    if getattr(backend, 'metadata', {}).get('model_version') == 'UNAVAILABLE':
        return backend.infer()
    return None


class InferenceWorker:
    def __init__(self):
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="p07-inference")
        self.future = None

    @property
    def busy(self):
        return self.future is not None

    def submit(self, backend, model, packet, expected, scenario, generation, previous=None):
        if self.busy:
            return False
        def run():
            start = time.perf_counter()
            first_prediction = (previous is not None and previous.get("_p07_needs_inference")
                                and expected == getattr(backend, "fixed", None))
            rejudge_only = previous is not None and not first_prediction
            try:
                if not rejudge_only:
                    backend.synchronize()
                    start = time.perf_counter()
                    result = backend.infer(packet.image, expected, scenario) if backend.is_mock else backend.infer(packet.image, expected)
                    backend.synchronize()
                    inference_ms = (time.perf_counter() - start) * 1000
                    if result.get("_p07_inference_executed") is False:
                        inference_ms = None
                else:
                    result = backend.rejudge(previous, expected)
                    inference_ms = None
                result = validate_result(result, model, packet.image.shape)
            except Exception as exc:
                result = dict(status="ERROR", predicted_label=None, prediction_score=None, score_type=None,
                              non_catalogue_count=None, counts=None, objects=[], message=str(exc))
                inference_ms = None
            result = dict(result, model_id=model, model_version=backend.metadata.get("model_version", "UNAVAILABLE"),
                          inference_time_ms=inference_ms, error_message=result.get("message") if result["status"] == "ERROR" else None)
            return dict(result=result, packet=packet, expected=dict(expected), model=model, generation=generation,
                        inference_ms=inference_ms, backend=backend, rejudge=rejudge_only)
        self.future = self.pool.submit(run)
        return True

    def poll(self):
        if self.future is not None and self.future.done():
            future, self.future = self.future, None
            return future.result()
        return None

    def close(self):
        self.pool.shutdown(wait=True)


def present(packet, result, expected, model, mode, info):
    """Draw presentation only; model results and contracts are not modified."""
    from .p07_dashboard import render
    return render(packet, result, expected, model, mode, info if isinstance(info, dict) else {})
