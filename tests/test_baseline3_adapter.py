"""Adapter/business-rule tests with explicit fixtures, not detector accuracy claims."""
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from src.adapters.baseline3 import Baseline3Backend
from src.models.baseline3.inference import _observations
from src.work_orders import reference_counts


class Baseline3AdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.config, self.weights = root/'config.json', root/'best.pt'
        self.config.write_text('{}')
        self.weights.write_bytes(b'interface fixture, not a model')
        self.rows = [[1.,1.,4.,4.,.8,float(cid)] for cid in range(1,6)]
        self.calls = 0
        def infer(frame, expected):
            self.calls += 1
            return _observations(self.rows,20,20)
        self.detector = SimpleNamespace(metadata=dict(weights=str(self.weights),model_version='FIXTURE'),
                                        _params=dict(max_det=300),synchronize=lambda:None,infer=infer)
        with patch('src.adapters.baseline3.load_detector',return_value=self.detector):
            self.backend = Baseline3Backend(self.config)
        self.frame = np.full((20,20,3),128,np.uint8)

    def test_count_rule_and_rejudge_preserve_all_observations(self):
        result=self.backend.infer(self.frame,reference_counts())
        self.assertEqual(result['status'],'PASS')
        original=copy.deepcopy(result)
        changed=self.backend.rejudge(result,dict.fromkeys('12345',0))
        self.assertEqual(changed['status'],'FAIL')
        self.assertEqual(changed['objects'],original['objects'])
        self.assertEqual(changed['counts'],original['counts'])
        self.assertEqual(self.calls,1)
        self.assertEqual(result,original)
        self.assertIsNone(changed['prediction_score'])
        self.assertTrue(np.all(self.frame==128))

    def test_learned_id0_forces_fail_and_warning_for_any_requirement(self):
        self.rows.append([5.,5.,8.,8.,.9,0.])
        result=self.backend.infer(self.frame,reference_counts())
        self.assertEqual(result['status'],'FAIL')
        self.assertIn('WARNING',result['message'])
        self.assertEqual(result['non_catalogue_count'],1)
        self.assertEqual(self.backend.rejudge(result,dict.fromkeys('12345',9))['status'],'FAIL')
        self.assertIn('NOT_PROVIDED',self.backend.metadata['size_verification'])

    def test_detection_cap_is_genuine_review_not_count_pass(self):
        self.detector._params['max_det']=len(self.rows)
        result=self.backend.infer(self.frame,reference_counts())
        self.assertEqual(result['status'],'REVIEW')
        self.assertIsNone(result['predicted_label'])

    def test_cap_does_not_hide_a_detected_id0_fail(self):
        self.rows.append([5.,5.,8.,8.,.9,0.])
        self.detector._params['max_det']=len(self.rows)
        result=self.backend.infer(self.frame,reference_counts())
        self.assertEqual(result['status'],'FAIL')
        self.assertIn('cap',result['message'])

    def test_changed_checkpoint_is_an_error_not_a_fallback(self):
        self.weights.write_bytes(b'changed')
        with self.assertRaisesRegex(RuntimeError,'reload'):
            self.backend.infer(self.frame,reference_counts())
