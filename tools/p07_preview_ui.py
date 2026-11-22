"""Render deterministic MOCK dashboard previews for visual layout review."""
import json
from pathlib import Path

import cv2
import numpy as np

from src.camera import Frame
from src.p07_bridge import MockBackend
from src.p07_dashboard import render


def main():
    config = json.loads(Path('config/p07.json').read_text())
    directory = Path('docs/p07/ui')
    directory.mkdir(parents=True, exist_ok=True)
    expected = config['expected_counts']
    image = np.full((720, 1280, 3), 220, np.uint8)
    cv2.putText(image, 'SYNTHETIC UI FIXTURE / NOT REAL DETECTIONS', (40, 640), cv2.FONT_HERSHEY_SIMPLEX, .8, (80, 80, 80), 2)
    packet = Frame(12, image, 0, 30, 'image:synthetic')
    for name, model, scenario in [('pass', 'baseline3', 'pass'), ('id0-fail', 'baseline3', 'id0'),
                                  ('review', 'final', 'review'), ('unsupported', 'baseline1', 'pass')]:
        counts = dict(expected, **{'1': 2}) if name == 'unsupported' else expected
        result = MockBackend(model, config).infer(image, counts, scenario)
        state = dict(source='image', camera_index=0, detecting=True, capture_fps=0, processing_fps=12.5,
                     inference_ms='14.20', processing_ms='25.30', resolution='1280x720', device='N/A',
                     message='MOCK layout preview; illustrative metrics, not measured performance')
        cv2.imencode('.png', render(packet, result, counts, model, 'mock', state))[1].tofile(directory/(name+'.png'))
    cv2.imencode('.png', render(None, None, expected, 'baseline3', 'mock', {}))[1].tofile(directory/'empty.png')


if __name__ == '__main__':
    main()
