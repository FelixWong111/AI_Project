"""Business-rule fixtures only. Synthetic trusted flags are NOT physical validation."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from src.adapters.baseline3 import SizeVerifiedBackend, create_backend
from src.models.baseline3.inference import _observations
from src.p07_bridge import ObjectBackend
from src.p07_output import export_result
from src.p07_dashboard import render
from src.camera import Frame


class SizeIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.expected = dict.fromkeys('12345', 1)
        rows = [[float(c*4),1.,float(c*4+2),5.,.8,float(c)] for c in range(1,6)]
        self.obs = _observations(rows,40,30)
        for obj in self.obs['objects']:
            obj['size_evidence'] = dict(conclusion='SPEC_COMPATIBLE_CANDIDATE',
                family_confirmed=True, compatible_class_ids=[obj['class_id']],
                measurement_policy_reason='VALIDATED_SCOPE_MATCHED')
        self.obs['p06_evidence'] = dict(raw_observations=_observations(rows,40,30),
            foreground_enabled=True, background_reason='', counts_complete=True, unresolved_candidates=[])
        self.backend = object.__new__(SizeVerifiedBackend)
        self.backend.files=();self.backend.signature=()
        self.backend.settings={'capture_geometry_id':'SYNTHETIC'}
        self.backend.approval_reason=None
        specs=json.loads(Path('config/p06_part_specs_candidate.json').read_text(encoding='utf-8'))
        self.backend.metadata=dict(business_decision_owner='baseline3_adapter',baseline3_mode='size_verified',
            calibration=dict(validated=True,capture_geometry_id='SYNTHETIC',frame_width=40,frame_height=30),
            size_tolerance=specs)
        self.calls=0
        def infer(frame, expected):
            self.calls+=1
            return copy.deepcopy(self.obs)
        self.backend.provider=SimpleNamespace(infer=infer,rejudge=lambda obs,counts:copy.deepcopy(obs),
                                             detector=SimpleNamespace(_params={'max_det':300}))
        self.frame=np.zeros((30,40,3),np.uint8)

    def infer(self):
        return ObjectBackend(self.backend,'baseline3').infer(self.frame,self.expected)

    def test_pass_requires_every_gate_not_blanket_review(self):
        self.assertEqual(self.infer()['status'],'PASS')
        self.backend.approval_reason='ACCEPTANCE_NOT_CONFIRMED'
        self.assertEqual(self.infer()['status'],'REVIEW')

    def test_missing_and_extra_counts_are_fail_even_without_validated_scale(self):
        self.backend.metadata['calibration']['validated']=False
        self.expected['1']=2
        self.assertEqual(self.infer()['status'],'FAIL')
        self.expected['1']=0
        self.assertEqual(self.infer()['status'],'FAIL')

    def test_id0_fail_precedes_unknown_size_and_truncation(self):
        obj=dict(detection_id=5,class_id=0,xmin=25.,ymin=1.,xmax=27.,ymax=4.,score=.7)
        self.obs['objects'].append(obj);self.obs['counts']['0']=1;self.obs['non_catalogue_count']=1
        self.backend.provider.detector._params['max_det']=5
        self.assertEqual(self.infer()['status'],'FAIL')

    def test_confirmed_wrong_size_not_overwritten_by_outer_count_rule(self):
        self.obs['objects'][0]['size_evidence']['conclusion']='OUTSIDE_FAMILY_CANDIDATE'
        result=self.infer()
        self.assertEqual(result['status'],'FAIL')
        self.assertIn('CONFIRMED_WRONG_SIZE',result['inspection']['failure_reasons'][0])
        self.assertEqual(result['counts']['0'],1)
        self.assertEqual(result['counts']['1'],0)
        self.assertEqual(result['objects'][0]['detector_class_id'],1)
        self.assertEqual(result['objects'][0]['class_id'],0)
        self.assertEqual(result['_baseline3_observation']['counts'],self.obs['counts'])
        changed=ObjectBackend(self.backend,'baseline3').rejudge(result,dict.fromkeys('12345',0))
        self.assertEqual(changed['counts'],result['counts'])
        self.assertEqual(changed['objects'],result['objects'])

    def test_unapproved_wrong_size_candidate_is_review_not_confirmed_rejection(self):
        self.obs['objects'][0]['size_evidence']['conclusion']='OUTSIDE_FAMILY_CANDIDATE'
        self.backend.approval_reason='ACCEPTANCE_NOT_CONFIRMED'
        self.assertEqual(self.infer()['status'],'REVIEW')

    def test_validated_family_conflict_is_fail_without_relabelling_yolo(self):
        self.obs['objects'][0]['size_evidence']['compatible_class_ids']=[2]
        result=self.infer();self.assertEqual(result['status'],'FAIL')
        self.assertEqual(result['objects'][0]['class_id'],1)

    def test_missing_failed_or_uncertain_measurement_is_review(self):
        for reason in ('CALIBRATION_MISSING','CROP_TOO_SMALL','UNMEASURABLE','BORDERLINE'):
            self.obs['objects'][0]['size_evidence']={'reason':reason}
            self.assertEqual(self.infer()['status'],'REVIEW')

    def test_calibration_resolution_geometry_and_validation_gates(self):
        for key,value in [('validated',False),('frame_width',80),('capture_geometry_id','other')]:
            old=self.backend.metadata['calibration'][key]
            self.backend.metadata['calibration'][key]=value
            self.assertEqual(self.infer()['status'],'REVIEW')
            self.backend.metadata['calibration'][key]=old
        self.backend.metadata['calibration']=None
        self.assertEqual(self.infer()['status'],'REVIEW')

    def test_foreground_disabled_missing_or_unresolved_cannot_pass(self):
        p=self.obs['p06_evidence']
        p['foreground_enabled']=False
        self.assertEqual(self.infer()['status'],'REVIEW')
        p['foreground_enabled']=True;p['background_reason']='BACKGROUND_REFERENCE_MISSING'
        self.assertEqual(self.infer()['status'],'REVIEW')
        p['background_reason']='';p['unresolved_candidates']=[{'class_id':None}]
        self.expected['1']=2
        self.assertEqual(self.infer()['status'],'REVIEW')

    def test_detection_cap_cannot_pass(self):
        self.backend.provider.detector._params['max_det']=5
        self.assertEqual(self.infer()['status'],'REVIEW')

    def test_change_counts_preserves_size_boxes_counts_without_reinfer(self):
        wrapper=ObjectBackend(self.backend,'baseline3');result=self.infer();saved=copy.deepcopy(result)
        changed=wrapper.rejudge(result,dict.fromkeys('12345',0))
        self.assertEqual(changed['status'],'FAIL')
        self.assertEqual(changed['objects'],result['objects']);self.assertEqual(changed['counts'],result['counts'])
        self.assertEqual(result,saved);self.assertEqual(self.calls,1)
        self.assertEqual(wrapper.rejudge(changed,self.expected)['status'],'PASS')

    def test_review_can_be_rejudged_to_count_failure(self):
        self.backend.approval_reason='ACCEPTANCE_NOT_CONFIRMED';r=self.infer()
        self.assertEqual(r['status'],'REVIEW')
        self.assertEqual(ObjectBackend(self.backend,'baseline3').rejudge(r,dict.fromkeys('12345',0))['status'],'FAIL')

    def test_snapshot_contract_and_full_diagnostic_sidecar(self):
        r=self.infer()
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'snapshot';export_result(out,'x.png','baseline3',r,self.expected,self.backend.metadata)
            data=json.loads((out/'diagnostics.json').read_text(encoding='utf-8'))
            self.assertEqual(data['images'][0]['result']['objects'],r['objects'])
            self.assertNotIn('_baseline3_observation',data['images'][0]['result'])
            self.assertTrue((out/'objects.csv').exists())

    def test_details_render_error_and_provisional_measurement(self):
        r=self.infer();r['objects'][0]['size_evidence']={'reason':'UNMEASURABLE'}
        canvas=render(Frame(1,self.frame,0,0,'image'),r,self.expected,'baseline3','real',
                      {'show_diagnostics':True,'diagnostic_page':-1})
        self.assertEqual(canvas.shape,(1000,1600,3))

    def test_invalid_mode_and_pipeline_error_have_no_frozen_fallback(self):
        with self.assertRaises(ValueError):create_backend(model_id='baseline3',config={'baseline3_mode':'typo'})
        with patch('src.p06_pipeline.create_backend',side_effect=RuntimeError('broken pipeline')),patch(
                'src.adapters.baseline3.load_detector',side_effect=AssertionError('fallback')):
            with self.assertRaisesRegex(RuntimeError,'broken pipeline'):
                create_backend(model_id='baseline3',config={'baseline3_mode':'size_verified'})

    def test_changed_files_and_corrupt_approval_are_errors(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'evidence';p.write_text('a')
            self.backend.files=(p,);self.backend.signature=self.backend._signature();p.write_text('changed')
            with self.assertRaisesRegex(RuntimeError,'reload'):self.infer()


if __name__=='__main__':unittest.main()
