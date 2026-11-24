"""Audit real artifacts; optionally verify inference on a supplied image. Never train/download."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from src.camera import Frame
from src.p07_bridge import image_spec, load_backend
from src.p07_ui import InferenceWorker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/p07.json"))
    parser.add_argument("--image", type=Path)
    parser.add_argument("--output", type=Path, default=Path("docs/p07/model-integration.json"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    packages = {name: importlib.metadata.version(name) for name in
                ("numpy", "opencv-python", "torch", "torchvision", "scikit-learn", "scikit-image", "joblib", "pandas", "Pillow", "scipy")}
    evidence = dict(timestamp=datetime.now(timezone.utc).isoformat(), python=platform.python_version(), packages=packages,
                    scope="real artifact audit; predictions only if supplied artifacts and image exist", models={})
    failed = False
    for model in ("baseline1", "baseline2"):
        record = dict(load="NOT_RUN", inference="BLOCKED", counts="NOT_RUN")
        evidence["models"][model] = record
        worker = None
        try:
            _, _, _, files = image_spec(model, config)
            record["artifacts"] = [dict(path=str(p), exists=p.is_file()) for p in files]
            backend = load_backend("real", model, config)
            record.update(load="PASS", metadata=backend.metadata)
            if args.image is None:
                record.update(inference="NOT_RUN", reason="No real input image supplied")
                failed = True
                continue
            frame = cv2.imdecode(np.fromfile(args.image, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("Image cannot be decoded: " + str(args.image))
            worker = InferenceWorker()
            worker.submit(backend, model, Frame(1, frame, time.perf_counter(), 0, "image:"+str(args.image)), backend.fixed, "pass", 0)
            result = None
            while result is None:
                result = worker.poll()
                time.sleep(.005)
            prediction = result["result"]
            record.update(inference="PASS" if prediction["status"] in ("PASS", "FAIL") else "FAIL",
                          prediction={k:v for k,v in prediction.items() if not k.startswith("_")}, input=str(args.image))
            changed = dict(backend.fixed)
            changed["1"] += 1
            unsupported = backend.infer(frame, changed)
            restored = backend.rejudge(backend.rejudge(prediction, changed), backend.fixed)
            assert unsupported["status"] == "UNSUPPORTED_COUNTS" and unsupported["prediction_score"] is None
            assert restored["predicted_label"] == prediction["predicted_label"]
            assert restored["prediction_score"] == prediction["prediction_score"]
            record["counts"] = "PASS"
            failed |= record["inference"] != "PASS"
        except Exception as exc:
            failed = True
            record.update(load="ERROR" if record["load"] != "PASS" else "PASS", error_message=str(exc))
        finally:
            if worker:
                worker.close()
        print(model, record["load"], record["inference"], record.get("error_message", ""))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
