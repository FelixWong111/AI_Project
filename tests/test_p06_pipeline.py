"""Integrated functional evidence using synthetic detections and images."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import cv2
import numpy as np

from src.p06_pipeline import EvidencePipeline,IntegratedBackend,create_backend
from src.p06_size_evidence import SizeEvidenceProcessor
from src.p06_foreground import ForegroundAssociator
from src.models.baseline3.inference import _observations
from src.p07_bridge import validate_result,load_backend

ROOT=Path(__file__).resolve().parents[1]


class Detector:
    metadata={'model_version':'SYNTHETIC_TEST_FIXTURE','output_capability':'object_level'}
    def __init__(self,rows):self.rows=rows;self.calls=0;self.syncs=0
    def infer(self,frame,expected):
        self.calls+=1
        return _observations(self.rows,frame.shape[1],frame.shape[0])
    def synchronize(self):self.syncs+=1


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.bg=np.full((200,300,3),240,np.uint8)
        self.frame=self.bg.copy()
        cv2.rectangle(self.frame,(50,50),(90,100),(30,)*3,-1)
        specs=json.loads((ROOT/'config/p06_part_specs_candidate.json').read_text(encoding='utf-8'))
        self.size=SizeEvidenceProcessor(specs)
        self.fg=ForegroundAssociator(self.bg,reference_id='SYNTHETIC',capture_geometry_id='fixture')
        self.pipe=EvidencePipeline(self.size,capture_geometry_id='fixture',foreground=self.fg)
        self.rows=[[48.,48.,93.,103.,.9,2]]
        self.expected=dict.fromkeys('12345',1)

    def backend(self,rows=None):
        self.detector=Detector(self.rows if rows is None else rows)
        return IntegratedBackend(self.detector,self.pipe,{'evidence':'SYNTHETIC'})

    def test_actual_association_and_size_guard_validate_p07(self):
        r=self.backend().infer(self.frame,self.expected)
        validate_result(r,'baseline3',self.frame.shape)
        self.assertEqual(r['objects'][0]['size_evidence']['reason'],'CALIBRATION_MISSING')
        self.assertEqual(r['counts']['2'],1)
        self.assertEqual(r['p06_evidence']['proposed_counts']['2'],1)

    def test_dedup_reindex_and_preserve_original_evidence(self):
        r=self.backend(self.rows+[[49.,49.,92.,102.,.8,2]]).infer(self.frame,self.expected)
        validate_result(r,'baseline3',self.frame.shape)
        self.assertEqual(r['counts']['2'],1)
        self.assertEqual(r['p06_evidence']['raw_observations']['counts']['2'],2)
        self.assertEqual(r['objects'][0]['source_detection_id'],0)
        self.assertEqual(r['objects'][0]['detection_id'],0)

    def test_cross_class_duplicates_and_unknown_foreground_stay_review(self):
        cv2.circle(self.frame,(220,110),20,(30,)*3,-1)
        r=self.backend(self.rows+[[49.,49.,92.,102.,.8,3]]).infer(self.frame,self.expected)
        self.assertEqual(len(r['objects']),2)
        self.assertFalse(r['p06_evidence']['counts_complete'])
        self.assertEqual(len(r['p06_evidence']['unresolved_candidates']),2)
        self.assertTrue(all(o['rejection_evidence']['conclusion']=='INSUFFICIENT_EVIDENCE' for o in r['objects']))

    def test_id0_is_distinguished_from_unmatched_region(self):
        r=self.backend([[48.,48.,93.,103.,.9,0]]).infer(self.frame,self.expected)
        self.assertEqual(r['objects'][0]['rejection_evidence']['conclusion'],'DETECTOR_NON_CATALOGUE_CANDIDATE')
        empty=self.backend([]).infer(self.frame,self.expected)
        self.assertEqual(empty['counts']['0'],0)
        self.assertEqual(len(empty['p06_evidence']['unresolved_candidates']),1)

    def test_no_background_is_explicit_and_raw_validation_still_runs(self):
        pipe=EvidencePipeline(self.size,capture_geometry_id='fixture')
        observations=_observations(self.rows,300,200)
        r=pipe.process(self.frame,observations)
        self.assertEqual(r['p06_evidence']['background_reason'],'BACKGROUND_REFERENCE_MISSING')
        observations['counts']['2']=4
        with self.assertRaises(ValueError):pipe.process(self.frame,observations)

    def test_rejudge_uses_cache_and_rejects_changed_evidence(self):
        backend=self.backend();r=backend.infer(self.frame,self.expected)
        with patch.object(self.pipe,'process',side_effect=AssertionError('remeasured')):
            new=backend.rejudge(r,dict.fromkeys('12345',0))
        self.assertEqual(self.detector.calls,1)
        self.assertEqual(r['objects'],new['objects'])
        new['objects'][0]['score']=.1
        with self.assertRaises(ValueError):backend.rejudge(new,self.expected)
        backend.synchronize();self.assertEqual(self.detector.syncs,1)

    def test_size_outside_family_becomes_rejection_evidence(self):
        result=dict(status='REVIEW',conclusion='OUTSIDE_FAMILY_CANDIDATE')
        original=self.size.process
        def enriched(*args,**kwargs):
            r=original(*args,**kwargs);r['objects'][0]['size_evidence']=copy.deepcopy(result);return r
        with patch.object(self.size,'process',side_effect=enriched):
            r=self.backend().infer(self.frame,self.expected)
        self.assertEqual(r['objects'][0]['rejection_evidence']['conclusion'],'SIZE_OUTSIDE_CONFIRMED_FAMILY_CANDIDATE')
        self.assertEqual(r['objects'][0]['class_id'],2)
        self.assertEqual(r['status'],'REVIEW')

    def test_factory_through_p07_loader(self):
        with patch('src.p06_pipeline.create_detector',return_value=Detector(self.rows)) as load:
            backend=load_backend('real','baseline3',{
                'bridge':'src.p06_pipeline:create_backend',
                'model_config':{'pipeline_config_path':'config/p06_pipeline_candidate.json'}})
        self.assertFalse(backend.is_mock)
        self.assertEqual(load.call_args.kwargs['model_id'],'baseline3')
        r=backend.infer(self.frame,self.expected)
        validate_result(r,'baseline3',self.frame.shape)
        self.assertEqual(r['objects'][0]['size_evidence']['reason'],'CALIBRATION_RESOLUTION_MISMATCH')

    def test_factory_rejects_bad_background_before_gpu_load(self):
        cfg=json.loads((ROOT/'config/p06_pipeline_candidate.json').read_text(encoding='utf-8'))
        with tempfile.TemporaryDirectory() as temp:
            image=Path(temp)/'background.png';cv2.imencode('.png',self.bg)[1].tofile(image)
            cfg['background']=dict(path=str(image),file_sha256='wrong',reference_id='x',roi=None)
            cfg['foreground_enabled']=True
            config=Path(temp)/'pipeline.json';config.write_text(json.dumps(cfg),encoding='utf-8')
            with patch('src.p06_pipeline.create_detector') as load:
                with self.assertRaisesRegex(ValueError,'SHA256'):create_backend(model_id='baseline3',config={'pipeline_config_path':str(config)})
                load.assert_not_called()

    def test_real_geometry_runs_inside_integrated_synthetic_chain(self):
        bg=np.full((500,700,3),240,np.uint8)
        frame=bg.copy()
        cv2.rectangle(frame,(100,140),(145,345),(25,)*3,-1)
        cv2.rectangle(frame,(145,200),(545,280),(25,)*3,-1)
        cfg=dict(calibration_id='SYNTHETIC_SCALE',capture_geometry_id='fixture',
                 frame_width=700,frame_height=500,px_per_mm=[20,20],validated=False)
        specs=json.loads((ROOT/'config/p06_part_specs_candidate.json').read_text(encoding='utf-8'))
        pipe=EvidencePipeline(SizeEvidenceProcessor(specs,cfg),capture_geometry_id='fixture',
                              foreground=ForegroundAssociator(bg,reference_id='SYNTHETIC',capture_geometry_id='fixture'))
        r=pipe.process(frame,_observations([[95.,135.,550.,350.,.9,1]],700,500))
        validate_result(r,'baseline3',frame.shape)
        size=r['objects'][0]['size_evidence']
        self.assertIn('thread_major_diameter',size['measurement']['dimensions'])
        self.assertEqual(size['conclusion'],'INSUFFICIENT_EVIDENCE')
