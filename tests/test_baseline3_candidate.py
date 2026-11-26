"""Candidate configuration and native-factory isolation; no held-out photos or model training."""
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.models.baseline3 import candidate as c
from src.models.baseline3.inference import PARAMETERS


class CandidateTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.weight=self.root/'fixture.pt';self.weight.write_bytes(b'synthetic checkpoint fixture')
        self.path=self.root/'config.json'
        self.config=dict(candidate_schema='yolo11n-candidate/v1',architecture='YOLO11n',model_id='baseline3',
            class_names={str(i):n for i,n in enumerate(c.CLASSES)},inference=dict(PARAMETERS,conf=.35,iou=.5),
            ultralytics_version='8.3.236',weights=str(self.weight),
            weights_sha256=hashlib.sha256(self.weight.read_bytes()).hexdigest(),candidate_version='synthetic-fixture',
            score_convention='uncalibrated detector score',rejection_scope='learned class0',initialization='synthetic fixture only')

    def save(self):
        self.path.write_text(json.dumps(self.config),encoding='utf-8')

    def test_selected_thresholds_load_without_changing_original_parameter_contract(self):
        self.save()
        runtime=SimpleNamespace(cuda=SimpleNamespace(get_device_name=lambda i:'SYNTHETIC'))
        model=SimpleNamespace(task='detect',names=dict(enumerate(c.CLASSES)),to=lambda d:None)
        before=dict(PARAMETERS)
        with patch.object(c,'_cuda_runtime',return_value=runtime),patch.object(c,'_load_model',return_value=model):
            backend=c.detector_factory(model_id='baseline3',config={'config_path':self.path})
        self.assertEqual(backend._params['conf'],.35)
        self.assertEqual(backend.metadata['model_version'],'synthetic-fixture-'+self.config['weights_sha256'][:12])
        self.assertEqual(PARAMETERS,before)

    def test_wrong_mapping_and_tampered_weights_fail_before_gpu_load(self):
        for bad_mapping in (True,False):
            with self.subTest(mapping=bad_mapping):
                if bad_mapping:self.config['class_names']['0']='flat_washer'
                else:
                    self.config['class_names']['0']='non_catalogue';self.weight.write_bytes(b'tampered')
                self.save()
                with patch.object(c,'_load_model',side_effect=AssertionError('must not load')):
                    with self.assertRaises(ValueError):c.detector_factory(model_id='baseline3',config={'config_path':self.path})

    def test_candidate_does_not_relax_resolution_augmentation_or_type_validation(self):
        for key,value in [('imgsz',960),('augment',True),('conf',float('nan')),('conf',True),('max_det',100)]:
            with self.subTest(key=key,value=value):
                self.config['inference']=dict(PARAMETERS);self.config['inference'][key]=value;self.save()
                with self.assertRaises(ValueError):c.detector_factory(model_id='baseline3',config={'config_path':self.path})

    def test_original_factory_still_refuses_candidate_thresholds(self):
        from src.models.baseline3.inference import create_backend
        self.config['schema_version']=1;self.save()
        with patch('src.models.baseline3.inference._load_model',side_effect=AssertionError('must reject before load')):
            with self.assertRaisesRegex(ValueError,'Frozen inference settings differ'):
                create_backend(model_id='baseline3',config={'config_path':self.path})

    def test_bridge_has_explicit_modes_without_fallback(self):
        with patch('src.adapters.baseline3.FrozenYOLOBackend') as factory:
            c.create_backend(model_id='baseline3',config={'baseline3_mode':'yolo_frozen','baseline3_config':str(self.path)})
            self.assertIs(factory.call_args.kwargs['detector_factory'],c.detector_factory)
        with patch('src.adapters.baseline3.SizeVerifiedBackend') as factory:
            c.create_backend(model_id='baseline3',config={'baseline3_mode':'size_verified','baseline3_pipeline_config':str(self.path)})
            self.assertTrue(callable(factory.call_args.kwargs['provider_factory']))
        with self.assertRaises(ValueError):c.create_backend(model_id='baseline3',config={'baseline3_mode':'unknown'})

    def test_display_timing_keeps_processing_timing_and_observed_throughput_separate(self):
        from scripts.benchmark_live import statistics
        rows=[dict(model='baseline3',phase='measure',processing_status='OK',processing_latency_ms=2,
                   live_latency_ms=10+i,available_perf=i*.05,display_done_perf=i*.05+.02,pixel_sha256=str(i)) for i in range(20)]
        s=statistics(rows,'baseline3',20,0)
        self.assertEqual(s['median_ms'],2);self.assertEqual(s['live_median_ms'],19.5)
        self.assertAlmostEqual(s['live_p95_ms'],28.05)
        self.assertAlmostEqual(s['observed_sample_fps'],20/.97)
        self.assertNotEqual(s['processing_fps'],s['observed_sample_fps'])

    def test_render_and_event_update_order(self):
        import numpy as np
        from scripts.benchmark_live import display_result
        calls=[];im=np.zeros((2,2,3),np.uint8)
        with patch('src.p07_ui.present',side_effect=lambda *a:(calls.append('render') or im)),patch(
                'cv2.imshow',side_effect=lambda *a:calls.append('imshow')),patch(
                'cv2.waitKey',side_effect=lambda *a:(calls.append('waitKey') or -1)):
            self.assertTrue(display_result(SimpleNamespace(image=im),{},dict.fromkeys('12345',1),'baseline3',{}))
        self.assertEqual(calls,['render','imshow','waitKey'])


if __name__=='__main__':unittest.main()
