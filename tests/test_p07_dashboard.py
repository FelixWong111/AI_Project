from p07_stubs import MockBackend
"""Exercise the real controller with mouse events; native display calls are stubbed."""
import csv
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

import app
from src.camera import Frame
from src.p07_bridge import MODELS
from src.p07_dashboard import GREEN, RED, AMBER, regions, hit_test, render
from tools.p07_demo import create
from p07_stubs import fixture_backend

CONFIG = json.loads(Path('config/p07.json').read_text())
COUNTS = dict.fromkeys('12345', 1)


class DashboardTests(unittest.TestCase):
    def test_controls_have_distinct_hit_regions(self):
        for action, (x1, y1, x2, y2) in regions():
            self.assertEqual(hit_test((x1+x2)//2, (y1+y2)//2), action)
        self.assertIsNone(hit_test(10, 700))

    def test_status_colors_and_fixed_recording_size(self):
        image = np.full((480, 640, 3), 128, np.uint8)
        original = image.copy()
        packet = Frame(1, image, 0, 0, 'image')
        for status, color in [('PASS', GREEN), ('FAIL', RED), ('REVIEW', AMBER),
                              ('UNSUPPORTED_COUNTS', AMBER), ('ERROR', RED)]:
            result = dict(status=status, objects=[], counts=None)
            canvas = render(packet, result, COUNTS, 'baseline1', 'mock', {})
            self.assertEqual(canvas.shape, (1000, 1600, 3))
            self.assertTrue(np.any(np.all(canvas[340:390, 1112:1570] == color, axis=2)))
        self.assertTrue(np.array_equal(image, original))

    def test_mouse_controls_reach_real_model_loader_and_exports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            create(root)
            callback = None
            ticks = 0
            loaded = []
            # Mouse events are sent through app's registered callback, not directly to handlers.
            schedule = {15: 'model:baseline1', 25: 'snapshot', 35: 'model:baseline2', 45: 'snapshot',
                        55: 'model:baseline3', 65: 'snapshot', 75: 'model:final', 85: 'snapshot',
                        95: 'count+:1', 105: 'snapshot', 115: 'toggle', 120: 'toggle',
                        125: 'record', 140: 'record', 150: 'quit'}
            def register(title, fn):
                nonlocal callback
                callback = fn
            def wait(delay):
                nonlocal ticks
                ticks += 1
                if ticks in schedule:
                    action = schedule[ticks]
                    rect = dict(regions())[action]
                    callback(cv2.EVENT_LBUTTONUP, (rect[0]+rect[2])//2, (rect[1]+rect[3])//2, 0, None)
                time.sleep(.005)
                return -1 if ticks < 180 else ord('q')
            def load(mode, model, config):
                loaded.append(model)
                return fixture_backend(model, config)
            with patch.object(app, 'load_backend', side_effect=load), patch.object(cv2, 'namedWindow'), \
                 patch.object(cv2, 'resizeWindow'), patch.object(cv2, 'setMouseCallback', side_effect=register), \
                 patch.object(cv2, 'imshow'), patch.object(cv2, 'waitKey', side_effect=wait), \
                 patch.object(cv2, 'getWindowProperty', return_value=1), patch.object(cv2, 'destroyAllWindows'):
                self.assertEqual(app.main(['--image', str(root/'mock-input.png'), '--output-dir', str(root/'out')]), 0)
            self.assertEqual(loaded, ['baseline1', *MODELS])
            exports = list((root/'out').rglob('images.csv'))
            models = set()
            for path in exports:
                with path.open() as stream:
                    models.add(next(csv.DictReader(stream))['model_id'])
            self.assertEqual(models, set(MODELS))
            self.assertEqual(len(exports), 5)
            changed = [json.loads(p.read_text()) for p in (root/'out').rglob('run_config.json')]
            self.assertTrue(any(next(iter(c['per_image_expected_counts'].values()))['1'] == 2 for c in changed))
            self.assertTrue(list((root/'out').rglob('*.mp4')))

    def test_missing_models_clear_old_predictions_and_reload_can_recover(self):
        """STUB_PASS for recovery; real missing-file errors use the actual loader."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            create(root)
            deployment = dict(CONFIG, model_config=dict(CONFIG['model_config'],
                              baseline1_bundle=str(root/'missing.joblib'), baseline2_bundle=str(root/'missing-bundle'),
                              baseline3_config=str(root/'missing-baseline3.json'),
                              baseline3_pipeline_config=str(root/'missing-pipeline.json')))
            config_path = root/'deployment.json'
            config_path.write_text(json.dumps(deployment), encoding='utf-8')
            callback, tick = None, 0
            scenes, loads = [], []
            schedule = {10: 'model:baseline1', 25: 'model:baseline2', 40: 'model:baseline3',
                        55: 'model:baseline1', 85: 'quit'}
            original_load, original_present = app.load_backend, app.present
            def load(mode, model, config):
                loads.append(model)
                # Simulate delivery on second baseline1 selection using a test-only stub.
                if model == 'baseline1' and loads.count(model) >= 3:
                    return fixture_backend(model, config)
                return original_load(mode, model, config)
            def render(packet, result, expected, model, mode, state):
                scenes.append((tick, model, result.get('status') if result else None, mode))
                return original_present(packet, result, expected, model, mode, state)
            def register(title, fn):
                nonlocal callback
                callback = fn
            def wait(delay):
                nonlocal tick
                tick += 1
                if tick in schedule:
                    x1,y1,x2,y2 = dict(regions())[schedule[tick]]
                    callback(cv2.EVENT_LBUTTONUP, (x1+x2)//2, (y1+y2)//2, 0, None)
                if tick == 70:
                    time.sleep(.005)
                    return ord('l')
                time.sleep(.005)
                return -1 if tick < 100 else ord('q')
            with patch.object(app,'load_backend',side_effect=load), patch.object(app,'present',side_effect=render), \
                 patch.object(cv2,'namedWindow'), patch.object(cv2,'resizeWindow'), \
                 patch.object(cv2,'setMouseCallback',side_effect=register), patch.object(cv2,'imshow'), \
                 patch.object(cv2,'waitKey',side_effect=wait), patch.object(cv2,'getWindowProperty',return_value=1), \
                 patch.object(cv2,'destroyAllWindows'):
                self.assertEqual(app.main(['--config',str(config_path),'--image',str(root/'mock-input.png'),'--output-dir',str(root/'out')]),0)
            self.assertTrue(any(m=='baseline1' and s=='ERROR' for _,m,s,_ in scenes))
            self.assertTrue(any(m=='baseline2' and s=='ERROR' for _,m,s,_ in scenes))
            self.assertFalse(any(m in ('baseline1','baseline2') and mode=='mock' for _,m,_,mode in scenes))
            self.assertFalse(any(m=='baseline2' and s in ('PASS','FAIL') for _,m,s,_ in scenes))
            self.assertTrue(any(t>55 and m=='baseline1' and s=='PASS' for t,m,s,_ in scenes))
            self.assertEqual(loads.count('baseline1'),4)  # explicit reload, not per-frame loading


if __name__ == '__main__':
    unittest.main()
