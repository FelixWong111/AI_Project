"""Requirements are inputs, never actual counts or ground-truth labels."""
import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np

from scripts.generate_work_orders import generate
from src.work_orders import load_cases, reference_counts, validate_counts, judge_observed
from src.p07_ui import current_result
from src.p07_bridge import ObjectBackend


class WorkOrderTests(unittest.TestCase):
    def test_generator_ignores_truth_and_rejects_guessing(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'manifest.json'
            p.write_text(json.dumps([dict(image_id='parts.jpg',split='validation',flat_washer_count=2,label='FAIL')]))
            result=generate(p,'validation')
            self.assertEqual(result,[dict(image_id='parts.jpg',expected_counts=reference_counts())])
            p.write_text(json.dumps([dict(image_id='TEST_01',split='validation')]))
            with self.assertRaisesRegex(ValueError,'extension'):generate(p,'validation')
            mapping=Path(tmp)/'mapping.csv';mapping.write_text('case_id,image_id\nTEST_01,parts.jpg\n')
            self.assertEqual(generate(p,'validation',mapping_csv=mapping),result)
            p.write_text(json.dumps([dict(image_id='parts.jpg'),dict(image_id='parts.jpg')]))
            with self.assertRaisesRegex(ValueError,'Duplicate'):generate(p)
            with self.assertRaisesRegex(ValueError,'fixed reference'):
                generate(p,counts=dict(reference_counts(),**{'5':2}))

    def test_shared_invalid_input_validation(self):
        fixed=reference_counts()
        for value in ({}, fixed|{'1':-1}, fixed|{'1':1.5}, fixed|{'1':True},fixed|{'0':0}):
            with self.assertRaises(ValueError):validate_counts(value)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'cases.json'
            p.write_text(json.dumps([dict(image_id='a.jpg',expected_counts=fixed,label='PASS')]))
            with self.assertRaisesRegex(ValueError,'ground truth'):load_cases(p)

    def test_gui_never_displays_obsolete_label_or_score(self):
        fixed=reference_counts();other=fixed|{'5':2}
        raw=dict(status='PASS',predicted_label='PASS',prediction_score=.99)
        cached=dict(model='baseline1',expected=fixed,result=raw)
        backend=SimpleNamespace(metadata={})
        self.assertEqual(current_result('baseline1',backend,fixed,cached),raw)
        unsupported=current_result('baseline1',backend,other,cached)
        self.assertEqual(unsupported['status'],'UNSUPPORTED_COUNTS')
        self.assertIsNone(unsupported['predicted_label']);self.assertIsNone(unsupported['prediction_score'])
        self.assertIsNone(current_result('baseline2',backend,fixed,cached))
        self.assertIsNone(current_result('baseline1',backend,fixed,cached,pending=True))
        self.assertEqual(current_result('baseline1',backend,fixed,cached),raw)
        self.assertEqual(cached['result']['predicted_label'],'PASS')

    def test_object_rejudge_never_calls_detector_or_alters_observation(self):
        # Interface fixture only, not evidence of a delivered baseline3 detector.
        class DetectorFixture:
            metadata={}
            def infer(self,*args):raise AssertionError('Count-only change must not run detection')
            def synchronize(self):pass
        backend=ObjectBackend(DetectorFixture(),'baseline3')
        raw=dict(status='PASS',predicted_label='PASS',prediction_score=None,
                 objects=[dict(class_id=5,xmin=1,ymin=2,xmax=3,ymax=4,score=.8)],
                 counts=dict.fromkeys('012345',0),non_catalogue_count=0)
        raw['counts']['5']=1;original=copy.deepcopy(raw)
        required=dict.fromkeys('12345',0)
        fail=backend.rejudge(raw,required)
        required['5']=1;passed=backend.rejudge(raw,required)
        self.assertEqual(fail['status'],'FAIL');self.assertEqual(passed['status'],'PASS')
        self.assertEqual(passed['counts'],original['counts']);self.assertEqual(passed['objects'],original['objects'])
        self.assertEqual(raw,original)
        raw['counts']['0']=1;raw['non_catalogue_count']=1
        self.assertEqual(backend.rejudge(raw,required)['status'],'FAIL')

    def test_object_inference_applies_id0_rule_before_validating_decision(self):
        class DetectorFixture:
            metadata={}
            def infer(self,*args):
                return dict(status='PASS',predicted_label='PASS',prediction_score=.9,score_type='prob',
                            counts=dict.fromkeys('12345',0)|{'0':1},non_catalogue_count=1,
                            objects=[dict(detection_id=0,class_id=0,xmin=0,ymin=0,xmax=2,ymax=2,score=.9)])
            def synchronize(self):pass
        result=ObjectBackend(DetectorFixture(),'baseline3').infer(np.zeros((3,3,3),np.uint8),dict.fromkeys('12345',0))
        self.assertEqual(result['status'],'FAIL')
        self.assertIsNone(result['prediction_score'])
        self.assertEqual(result['objects'][0]['score'],.9)
