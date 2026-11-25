"""Synthetic scopes test wiring only. No actual error bounds are certified."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from src.p06_measurement_policy import MeasurementPolicy,load_policy,GEOMETRY
from src.p06_geometry import DEFAULT_PARAMETERS
from src.p06_pipeline import create_backend
from src.p06_inference import _observations


ROOT=Path(__file__).resolve().parents[1]


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.record=dict(evidence_id='SYNTHETIC_SCOPE',status='VALIDATED_MEASUREMENT_SCOPE',
                         geometry_sha256=hashlib.sha256(GEOMETRY.read_bytes()).hexdigest(),
                         parameters=DEFAULT_PARAMETERS,calibration_id='fixture',capture_geometry_id='fixture',
                         frame_width=100,frame_height=80,class_ids=[1,2,3],
                         dimensions={'thread_major_diameter':.1},
                         metric_ranges={'shaft_line_fraction':[.25,.75]},family_confirmed=True,px_per_mm=[20,20])
        self.cal=dict(calibration_id='fixture',capture_geometry_id='fixture',frame_width=100,frame_height=80,
                      px_per_mm=[20,20],validated=True)
        self.measure=dict(class_id=2,status='PARTIAL_CANDIDATE',dimensions={'thread_major_diameter':{'mm_estimate':6}},
                          diagnostics={'shaft_line_fraction':.5})

    def test_scope_and_error_bound(self):
        policy=MeasurementPolicy([self.record])
        quality,family,reason=policy.assess(self.measure,self.cal,'fixture',(100,80))
        self.assertTrue(family)
        self.assertEqual(quality['thread_major_diameter']['max_error_mm'],.1)
        self.assertEqual(reason,'VALIDATED_SCOPE_MATCHED')
        for cal,geo,size in [(dict(self.cal,validated=False),'fixture',(100,80)),
                             (dict(self.cal,calibration_id='other'),'fixture',(100,80)),
                             (dict(self.cal,px_per_mm=[30,30]),'fixture',(100,80)),
                             (self.cal,'other',(100,80)),(self.cal,'fixture',(200,80))]:
            self.assertEqual(policy.assess(self.measure,cal,geo,size)[0],{})
        self.measure['diagnostics']['shaft_line_fraction']=.9
        self.assertEqual(policy.assess(self.measure,self.cal,'fixture',(100,80))[0],{})

    def test_no_evidence_and_conflicting_evidence(self):
        self.assertEqual(MeasurementPolicy().assess(self.measure,self.cal,'fixture',(100,80))[0],{})
        self.assertEqual(MeasurementPolicy([self.record,self.record]).assess(
            self.measure,self.cal,'fixture',(100,80))[2],'AMBIGUOUS_VALIDATION_SCOPES')
        with self.assertRaises(ValueError):MeasurementPolicy([dict(self.record,geometry_sha256='changed')])

    def test_factory_wires_scope_and_disables_foreground(self):
        cfg=json.loads((ROOT/'config/p06_pipeline_candidate.json').read_bytes())
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            evidence=folder/'record.json';evidence.write_text(json.dumps(self.record),encoding='utf-8')
            registry=folder/'registry.json'
            registry.write_text(json.dumps(dict(schema_version=1,records=[dict(path=str(evidence),
                sha256=hashlib.sha256(evidence.read_bytes()).hexdigest())])),encoding='utf-8')
            cal=folder/'calibration.json';cal.write_text(json.dumps(self.cal),encoding='utf-8')
            cfg.update(measurement_policy_path=str(registry),calibration_path=str(cal),capture_geometry_id='fixture',
                       background={'deliberately_invalid':'disabled module must not read it'})
            config=folder/'pipeline.json';config.write_text(json.dumps(cfg),encoding='utf-8')
            class Detector:
                metadata={}
                def infer(inner,frame,expected):return _observations([[10.,10.,50.,50.,.9,2]],100,80)
            with patch('src.p06_pipeline.create_detector',return_value=Detector()),patch(
                    'src.p06_pipeline.ForegroundAssociator',side_effect=AssertionError('foreground ran')):
                backend=create_backend(model_id='baseline3',config={'pipeline_config_path':str(config)})
            with patch('src.p06_size_evidence.measure_object',return_value=self.measure):
                result=backend.infer(np.zeros((80,100,3),np.uint8),dict.fromkeys('12345',1))
            obj=result['objects'][0]
            self.assertEqual(obj['size_evidence']['measurement_policy_reason'],'VALIDATED_SCOPE_MATCHED')
            self.assertEqual(obj['rejection_evidence']['conclusion'],'SIZE_OUTSIDE_CONFIRMED_FAMILY_CANDIDATE')
            self.assertFalse(result['p06_evidence']['foreground_enabled'])
            self.assertEqual(result['p06_evidence']['background_reason'],'FOREGROUND_DISABLED')
            self.assertEqual(result['counts']['2'],1)
            evidence.write_text('{}',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'SHA256'):load_policy(registry,ROOT)
