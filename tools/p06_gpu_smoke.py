"""Real CUDA + official YOLO11n smoke only. No training or project-class metrics."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', required=True, type=Path, help='Local official yolo11n.pt, outside Git or ignored')
    parser.add_argument('--cache', required=True, type=Path, help='Ignored process-local settings and synthetic image directory')
    parser.add_argument('--output', required=True, type=Path, help='New JSON evidence record; will not overwrite')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output already exists')
    args.cache.mkdir(parents=True, exist_ok=True)
    # Child-process settings only: no persistent user/system environment changes.
    settings_dir = (args.cache / 'ultralytics-settings').resolve()
    settings_dir.mkdir(parents=True, exist_ok=True)
    os.environ['YOLO_CONFIG_DIR'] = str(settings_dir)
    os.environ['TORCH_HOME'] = str((args.cache / 'torch').resolve())
    report = dict(timestamp=datetime.now(timezone.utc).isoformat(), status='FAIL',
                  python=sys.version, executable=sys.executable,
                  purpose='COCO YOLO11n CUDA smoke on a synthetic image; NOT mechanical component recognition',
                  weights=str(args.weights.resolve()), versions={}, checks={}, errors=[])
    try:
        import numpy as np
        import cv2
        import torch
        from ultralytics import YOLO
        from ultralytics.utils import USER_CONFIG_DIR

        report['settings_directory'] = str(USER_CONFIG_DIR)
        if args.cache.resolve() not in Path(USER_CONFIG_DIR).resolve().parents:
            raise RuntimeError('Ultralytics settings escaped the intended ignored cache directory')

        for package in ('numpy', 'opencv-python', 'cv2-enumerate-cameras', 'torch', 'torchvision', 'ultralytics', 'Pillow', 'pip'):
            report['versions'][package] = importlib.metadata.version(package)
        pip = subprocess.run([sys.executable, '-m', 'pip', 'check'], capture_output=True, text=True)
        report['pip_check'] = dict(exit_code=pip.returncode, stdout=pip.stdout, stderr=pip.stderr)
        if pip.returncode:
            raise RuntimeError('pip check failed')
        expected = {'numpy':'1.26.4', 'opencv-python':'4.10.0.84', 'cv2-enumerate-cameras':'1.4.0',
                    'torch':'2.7.1+cu126', 'torchvision':'0.22.1+cu126', 'ultralytics':'8.3.236'}
        if any(report['versions'][name] != version for name, version in expected.items()):
            raise RuntimeError(f'Unexpected dependency versions: {report["versions"]}')
        if sys.version_info[:2] != (3,11):
            raise RuntimeError('Python 3.11 required')
        report['cuda_runtime'] = torch.version.cuda
        report['cuda_available'] = torch.cuda.is_available()
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; CPU fallback forbidden')
        report['gpu_name'] = torch.cuda.get_device_name(0)
        x = torch.tensor([1., 2., 3.], device='cuda:0')
        value = (x*x).sum()
        torch.cuda.synchronize(0)
        report['tensor'] = dict(input_device=str(x.device), result_device=str(value.device),
                                expression='sum([1,2,3] ** 2)', result=value.item())
        if x.device.type != 'cuda' or value.device.type != 'cuda' or value.item() != 14.0:
            raise RuntimeError('CUDA tensor verification failed')
        report['checks']['cuda_tensor'] = 'PASS'
        # Independent synthetic geometry check, explicitly not a model prediction.
        from ultralytics.utils.ops import scale_boxes
        letterboxed = torch.tensor([[100.,208.,250.,308.]], device='cuda:0')
        restored = scale_boxes((640,640), letterboxed.clone(), (384,640))
        expected_box = torch.tensor([[100.,80.,250.,180.]], device='cuda:0')
        if not torch.allclose(restored, expected_box, atol=1e-5, rtol=0):
            raise RuntimeError('Ultralytics original-image coordinate restoration failed')
        report['synthetic_geometry_check'] = dict(source_shape=[384,640], letterbox_shape=[640,640],
                                                 restored_xyxy=restored.cpu().tolist(),
                                                 evidence='Known synthetic box, NOT an inferred detection', status='PASS')
        if args.weights.name != 'yolo11n.pt' or not args.weights.is_file():
            raise RuntimeError('Provide the downloaded official yolo11n.pt first')
        report['weights_sha256'] = hashlib.sha256(args.weights.read_bytes()).hexdigest()
        report['weights_bytes'] = args.weights.stat().st_size
        report['weights_source'] = 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt'
        model = YOLO(str(args.weights))
        model.to('cuda:0')
        report['checks']['model_loaded'] = 'PASS'
        report['class_names'] = model.names
        report['model_task'] = model.task
        if model.task != 'detect' or len(model.names) != 80 or model.names[0] != 'person':
            raise RuntimeError('Expected the official 80-class COCO detection checkpoint, not a project model')
        forward_devices = []

        def capture_device(module, inputs):
            forward_devices.append(str(inputs[0].device))

        hook = model.model.register_forward_pre_hook(capture_device)
        image = np.full((384, 640, 3), 255, dtype=np.uint8)
        cv2.rectangle(image, (100,80), (250,180), (80,80,80), -1)
        if not cv2.imwrite(str(args.cache / 'synthetic-smoke.png'), image):
            raise RuntimeError('Could not save synthetic input image')
        runs = []
        try:
            # Second real inference uses a maximum threshold to exercise empty-result handling.
            for confidence in (0.25, 1.0):
                result = model.predict(source=image, device=0, imgsz=640, conf=confidence,
                                       save=False, verbose=False)[0]
                torch.cuda.synchronize(0)
                boxes = result.boxes
                if boxes is None:
                    raise RuntimeError('Missing Boxes result structure')
                xyxy = boxes.xyxy.detach().cpu().numpy()
                classes = boxes.cls.detach().cpu().numpy()
                scores = boxes.conf.detach().cpu().numpy()
                if tuple(result.orig_shape) != image.shape[:2] or xyxy.shape != (len(boxes),4):
                    raise RuntimeError('Unexpected original-image coordinate/result shape')
                if classes.shape != (len(boxes),) or scores.shape != (len(boxes),):
                    raise RuntimeError('Unexpected class/confidence shapes')
                if not all(np.isfinite(a).all() for a in (xyxy, classes, scores)):
                    raise RuntimeError('NaN/Inf in inference output')
                if len(boxes) and (not ((xyxy[:,0] >= 0).all() and (xyxy[:,1] >= 0).all()
                                       and (xyxy[:,2] <= 640).all() and (xyxy[:,3] <= 384).all()
                                       and (xyxy[:,2] > xyxy[:,0]).all() and (xyxy[:,3] > xyxy[:,1]).all())
                                   or not ((scores >= 0).all() and (scores <= 1).all())
                                   or not ((classes >= 0).all() and (classes < len(model.names)).all()
                                           and (classes == np.floor(classes)).all())):
                    raise RuntimeError('Invalid original-image box/class/confidence values')
                parameter_device = str(next(model.model.parameters()).device)
                if parameter_device != 'cuda:0' or boxes.data.device.type != 'cuda':
                    raise RuntimeError('Inference model or Boxes data unexpectedly ran on CPU')
                runs.append(dict(conf=confidence, original_shape=list(result.orig_shape),
                                 model_device=parameter_device, boxes_device=str(boxes.data.device),
                                 xyxy_shape=list(xyxy.shape), classes_shape=list(classes.shape),
                                 confidence_shape=list(scores.shape), detection_count=len(boxes),
                                 boxes_xyxy=xyxy.tolist(), classes=classes.tolist(), confidence=scores.tolist(),
                                 finite=True, original_coordinate_bounds='PASS'))
        finally:
            hook.remove()
        if not forward_devices or any(device != 'cuda:0' for device in forward_devices):
            raise RuntimeError(f'Forward input device verification failed: {forward_devices}')
        if runs[1]['detection_count'] != 0:
            raise RuntimeError('Empty-result branch not exercised; do not claim PASS')
        report.update(forward_input_devices=forward_devices, inference_runs=runs, status='PASS')
        report['checks'].update(gpu_forward='PASS', boxes_classes_confidence='PASS',
                                original_coordinates='PASS', finite_outputs='PASS', empty_results='PASS')
    except Exception as exc:
        report['errors'].append(f'{type(exc).__name__}: {exc}')
        report['traceback'] = traceback.format_exc()
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
