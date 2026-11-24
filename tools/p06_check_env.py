"""Read-only dependency/CUDA probe; never installs or falls back to CPU."""
import argparse
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
import sys


def probe():
    report = dict(timestamp=datetime.now(timezone.utc).isoformat(), python=sys.version,
                  executable=sys.executable, python311=sys.version_info[:2] == (3,11),
                  versions={}, errors=[], cuda={'status': 'NOT_RUN'}, yolo11n='NOT_RUN')
    modules = {}
    for name in ('numpy', 'cv2', 'ultralytics', 'torch', 'torchvision'):
        try:
            module = importlib.import_module(name)
            modules[name] = module
            report['versions'][name] = module.__version__
        except Exception as exc:
            report['errors'].append(f'{name}: {type(exc).__name__}: {exc}')
    torch = modules.get('torch')
    if torch is not None:
        report['cuda'] = dict(status='FAIL', built_cuda=torch.version.cuda,
                              available=torch.cuda.is_available(), device_name=None, tensor_result=None)
        try:
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA is unavailable; GPU tensor test NOT_RUN; CPU fallback forbidden.')
            report['cuda']['device_name'] = torch.cuda.get_device_name(0)
            x = torch.tensor([1.0, 2.0, 3.0], device='cuda:0')
            value = (x*x).sum().item()
            torch.cuda.synchronize(0)
            if x.device.type != 'cuda' or value != 14.0:
                raise RuntimeError('CUDA tensor calculation returned an unexpected result.')
            report['cuda'].update(status='PASS', tensor_result=value)
        except Exception as exc:
            report['errors'].append(f'CUDA: {type(exc).__name__}: {exc}')
    if not report['python311']:
        report['errors'].append('Required Python 3.11 interpreter is unavailable in this invocation.')
    report['status'] = 'PASS' if not report['errors'] and report['cuda']['status'] == 'PASS' else 'FAIL'
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='New JSON record; existing file is refused')
    args = parser.parse_args()
    report = probe()
    if args.output:
        with args.output.open('x', encoding='utf-8') as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
