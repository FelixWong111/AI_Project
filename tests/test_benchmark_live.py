"""No real camera/model benchmarks: isolated fixtures, reports deleted with tempdir."""
import csv
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from scripts import benchmark_live as b


class BenchmarkTests(unittest.TestCase):
    def test_statistics_include_fail_review_exclude_warmup_errors(self):
        rows=[]
        for value,status in [(1,'PASS'),(2,'FAIL'),(3,'REVIEW')]:
            rows.append(dict(model='baseline1',phase='measure',processing_status='OK',final_status=status,
                             processing_latency_ms=value,pixel_sha256=str(value)))
        rows += [dict(model='baseline1',phase='warmup',processing_status='OK',processing_latency_ms=999),
                 dict(model='baseline1',phase='measure',processing_status='ERROR',processing_latency_ms=999),
                 dict(model='baseline1',phase='measure',processing_status='CAPTURE_ERROR')]
        s=b.statistics(rows,'baseline1',4,1)
        self.assertEqual(s['median_ms'],2);self.assertAlmostEqual(s['p95_ms'],2.9)
        self.assertEqual(s['coverage'],3/5);self.assertEqual(s['failed_frames'],2)
        self.assertEqual(s['processing_coverage'],.75);self.assertEqual(s['minimum_20_status'],'INSUFFICIENT')
        self.assertEqual(s['processing_fps'],500)

    def test_gpu_sync_boundaries_and_complete_infer(self):
        calls=[]
        backend=SimpleNamespace(synchronize=lambda:calls.append('sync'),infer=lambda im,c:(calls.append('infer') or
          dict(status='REVIEW',predicted_label=None,prediction_score=None,score_type='none',objects=[],counts=None,non_catalogue_count=None)))
        def clock():calls.append('clock');return 1 if calls.count('clock')==1 else 1.25
        with patch.object(b.time,'perf_counter',side_effect=clock):
            result,ms=b.process(backend,SimpleNamespace(image=np.zeros((2,2,3),np.uint8)),{},'baseline1')
        self.assertEqual(calls,['sync','clock','infer','sync','clock']);self.assertEqual(ms,250)
        self.assertEqual(result['status'],'REVIEW')

    def test_freshness_rejects_old_packet(self):
        old=SimpleNamespace(frame_id=1,available_at=1)
        provider=SimpleNamespace(latest=old,finished=True)
        self.assertIsNone(b.fresh_frame(provider,2,0,.01))
        self.assertIsNone(b.fresh_frame(provider,0,1,.01))

    def test_minimum_twenty_and_identical_stream_warning(self):
        rows=[dict(model='baseline1',phase='measure',processing_status='OK',processing_latency_ms=2,
                   pixel_sha256=str(i)) for i in range(20)]
        self.assertEqual(b.statistics(rows,'baseline1',20,0)['minimum_20_status'],'MET_NEW_READ_PROTOCOL')
        for r in rows:r['pixel_sha256']='same'
        self.assertEqual(b.statistics(rows,'baseline1',20,0)['minimum_20_status'],'NOT_VERIFIED_IDENTICAL_PIXELS')

    def test_cli_exports_and_camera_release_in_temporary_workspace(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);cfg=root/'cfg.json';cfg.write_text(json.dumps({'camera_backend':'auto','width':2,'height':2}))
            cameras=[];calls=[]
            class Camera:
                def __init__(self,kind,value,config,journal):
                    self.journal=journal;self.finished=False;self.error='';self.closed=False;cameras.append(self)
                def start(self):return self
                @property
                def latest(self):
                    image=np.full((2,2,3),len(calls),np.uint8)
                    fid=self.journal.capture('camera',0,'SYNTHETIC_FIXTURE',image,1,30,True)
                    return SimpleNamespace(frame_id=fid,image=image,available_at=time.perf_counter())
                def close(self):self.closed=True
            backend=SimpleNamespace(metadata={'model':'baseline1'},synchronize=lambda:None,
                 infer=lambda image,counts:(calls.append(1) or dict(status='FAIL',predicted_label='FAIL',
                     prediction_score=.9,score_type='prob',objects=[],counts=None,non_catalogue_count=None)))
            with patch.object(b,'ROOT',root),patch.object(b,'FrameProvider',Camera),patch.object(b,'load_backend',return_value=backend),patch.object(
                    b,'environment',return_value={'cpu':'SYNTHETIC','gpu':[],'dependencies':{}}):
                code=b.main(['--config',str(cfg),'--models','baseline1','--frames','2','--warmup','1'])
            self.assertEqual(code,1)  # Does not meet 20; fixture data must never qualify as real evidence.
            self.assertTrue(cameras[0].closed);self.assertEqual(len(calls),3)
            out=next((root/'results/live_benchmark').iterdir())
            self.assertTrue(all((out/n).exists() for n in ['frame_latency.csv','summary.csv','run_config.json','report.md','capture_events.csv']))
            with (out/'summary.csv').open(encoding='utf-8-sig') as f:s=next(csv.DictReader(f))
            self.assertEqual(s['successful_frames'],'2')

    def test_reject_yolo_only_before_environment_or_capture(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'cfg.json';p.write_text(json.dumps({'model_config':{'baseline3_mode':'yolo_frozen'}}))
            with patch.object(b,'environment',side_effect=AssertionError('hardware queried')):
                with self.assertRaises(SystemExit):b.main(['--config',str(p),'--models','baseline3'])

    def test_capture_failure_and_interrupt_export_partial_results_and_release(self):
        for interrupt in (False,True):
            with self.subTest(interrupt=interrupt),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);cfg=root/'cfg.json';cfg.write_text('{}');cameras=[]
                class BrokenCamera:
                    finished=True;error='SYNTHETIC read failure'
                    def __init__(self,kind,value,config,journal):self.journal=journal;self.closed=False;cameras.append(self)
                    def start(self):self.journal.capture('camera',0,'SYNTHETIC',None,1,0,False);return self
                    @property
                    def latest(self):
                        if interrupt:raise KeyboardInterrupt
                        return None
                    def close(self):self.closed=True
                with patch.object(b,'ROOT',root),patch.object(b,'FrameProvider',BrokenCamera),patch.object(
                    b,'load_backend',return_value=SimpleNamespace(metadata={})),patch.object(
                    b,'environment',return_value={'cpu':'SYNTHETIC','gpu':[],'dependencies':{}}):
                    self.assertEqual(b.main(['--config',str(cfg),'--models','baseline1','--frames','1','--warmup','1']),1)
                self.assertTrue(cameras[0].closed)
                out=next((root/'results/live_benchmark').iterdir())
                config=json.loads((out/'run_config.json').read_text(encoding='utf-8'))
                self.assertEqual(config['run_status'],'INTERRUPTED' if interrupt else 'COMPLETED_WITH_ERRORS')
                with (out/'frame_latency.csv').open(encoding='utf-8-sig') as stream:
                    self.assertTrue(any(r['processing_status']=='CAPTURE_ERROR' for r in csv.DictReader(stream)))


if __name__=='__main__':unittest.main()
