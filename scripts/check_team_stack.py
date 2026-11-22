"""Run after installation on the team's machine; no weights are downloaded."""

import argparse
from importlib import metadata
import json
from pathlib import Path
import platform
import subprocess
import sys


EXPECTED = {
    "numpy": "1.26.4", "opencv-python": "4.10.0.84",
    "cv2-enumerate-cameras": "1.4.0", "torch": "2.7.1",
    "torchvision": "0.22.1", "ultralytics": "8.3.236",
    "scipy": "1.15.3", "scikit-learn": "1.9.1", "Pillow": "13.0.0",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="cpu or cuda:0")
    parser.add_argument("--output", type=Path, default=Path("output/team_stack_check.json"))
    args = parser.parse_args()
    report = {
        "scope": "environment and synthetic execution only; no course accuracy validation",
        "python": platform.python_version(), "executable": sys.executable,
        "device": args.device, "versions": {}, "checks": {},
        "formal_yolo_weights": "NOT_RUN", "course_evaluation": "NOT_RUN",
    }

    def check(name, action):
        try:
            detail = action()
            report["checks"][name] = {"status": "PASS", "detail": detail}
        except Exception as error:
            report["checks"][name] = {"status": "FAIL", "error": str(error)}

    def require(condition, message):
        if not condition:
            raise RuntimeError(message)

    check("python_version", lambda: require(
        report["python"] == "3.11.14", "Expected Python 3.11.14"))
    for package, target in EXPECTED.items():
        def version_check(package=package, target=target):
            actual = metadata.version(package)
            report["versions"][package] = actual
            # Official CPU/CUDA wheels use local suffixes such as +cpu/+cu126.
            require(actual.split("+", 1)[0] == target,
                    f"{package}: expected {target}, installed {actual}")
        check(f"version:{package}", version_check)

    def dependency_check():
        result = subprocess.run([sys.executable, "-m", "pip", "check"],
                                capture_output=True, text=True)
        detail = (result.stdout + result.stderr).strip()
        require(result.returncode == 0, detail)
        return detail
    check("pip_check", dependency_check)

    def torch_check():
        import torch
        from torchvision.models import resnet18
        from torchvision.ops import nms

        device = torch.device(args.device)
        require(device.type in {"cpu", "cuda"}, "Use cpu or a CUDA device")
        if device.type == "cuda":
            require(torch.cuda.is_available(), "CUDA is unavailable")
        torch.manual_seed(42)
        model = resnet18(weights=None).eval().to(device)
        model.fc = torch.nn.Identity()
        with torch.inference_mode():
            features = model(torch.zeros((1, 3, 224, 224), device=device))
        require(tuple(features.shape) == (1, 512) and bool(torch.isfinite(features).all()),
                "ResNet18 feature output is invalid")
        boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 9., 9.]], device=device)
        scores = torch.tensor([.9, .8], device=device)
        require(nms(boxes, scores, .5).tolist() == [0], "TorchVision NMS failed")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return {"torch_cuda_build": torch.version.cuda, "feature_shape": [1, 512]}
    check("resnet18_and_nms", torch_check)

    def clahe_check():
        import numpy as np
        from src.stage3_clahe import DetectorCLAHE

        frame = np.random.default_rng(42).integers(0, 100, (64, 96, 3), dtype=np.uint8)
        original = frame.copy()
        enhanced = DetectorCLAHE(2.0, (8, 8))(frame)
        require(enhanced.shape == frame.shape and enhanced.dtype == frame.dtype,
                "CLAHE changed geometry or dtype")
        require(np.array_equal(frame, original), "CLAHE modified the input frame")
        return "BGR uint8 shape and original frame preserved"
    check("clahe", clahe_check)

    def yolo_check():
        import numpy as np
        from ultralytics import YOLO

        # YAML constructs random weights locally, rather than downloading a .pt file.
        model = YOLO("yolo11n.yaml", task="detect")
        frame = np.zeros((64, 96, 3), dtype=np.uint8)
        results = model.predict(source=frame, device=args.device, imgsz=64,
                                conf=.99, max_det=10, save=False, verbose=False)
        require(len(results) == 1 and results[0].orig_shape == (64, 96),
                "YOLO11n did not preserve original-image metadata")
        require(results[0].boxes is not None, "YOLO11n detection output is missing")
        return "Random-initialized YOLO11n executed; no trained recognition claim"
    check("yolo11n_random_inference", yolo_check)

    report["status"] = ("PASS" if all(c["status"] == "PASS"
                                     for c in report["checks"].values()) else "FAIL")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(f"{report['status']}: {args.output}")
    for name, result in report["checks"].items():
        if result["status"] == "FAIL":
            print(f"  {name}: {result['error']}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
