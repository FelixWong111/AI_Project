"""Synthetic interface tests; these doubles are not model performance evidence."""
import copy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from src import p06_inference as inference
from src.p07_bridge import load_backend,validate_result

class FakeData:
    is_cuda=True
    def __init__(self,rows):self.rows=rows
    def cpu(self):return self
    def tolist(self):return copy.deepcopy(self.rows)

class FakeEngine:
    names=dict(enumerate(inference.CLASSES))
    task='detect'
    def __init__(self,rows):
        self.rows=rows;self.calls=[]
        self.model=SimpleNamespace(parameters=lambda:iter([SimpleNamespace(device=SimpleNamespace(type='cuda',index=0))]))
    def to(self,device):self.device=device;return self
    def predict(self,**kwargs):
        self.calls.append(kwargs)
        frame=kwargs['source'];shape=frame.shape[:2]
        # Deliberate mutation proves the caller's frame is isolated.
        frame[:]=0
        return [SimpleNamespace(orig_shape=shape,boxes=SimpleNamespace(data=FakeData(self.rows)))]

class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.folder=Path(self.tmp.name)
        self.cfg=json.loads((inference.ROOT/'config/p06_baseline3.json').read_text(encoding='utf-8'))
        self.weight=self.folder/'fixture.pt';self.weight.write_bytes(b'synthetic checkpoint fixture, never a real model')
        self.cfg['weights']=str(self.weight);self.cfg['weights_sha256']=hashlib.sha256(self.weight.read_bytes()).hexdigest()
        self.path=self.folder/'config.json';self.path.write_text(json.dumps(self.cfg),encoding='utf-8')
        self.engine=FakeEngine([[10.,5.,40.,35.,.8,2.]])
        self.sync=[]
        cuda=SimpleNamespace(synchronize=lambda device:self.sync.append(device),get_device_name=lambda device:'Synthetic GPU fixture')
        self.runtime=SimpleNamespace(cuda=cuda)
        self.expected={str(i):1 for i in range(1,6)}
    def make(self):
        with patch.object(inference,'_cuda_runtime',return_value=self.runtime),patch.object(inference,'_load_model',return_value=self.engine):
            return inference.create_backend(model_id='baseline3',config={'config_path':str(self.path)})
    def test_six_counts_original_coords_and_input_not_mutated(self):
        backend=self.make();frame=np.full((40,80,3),120,np.uint8)
        result=backend.infer(frame,self.expected)
        self.assertTrue(np.all(frame==120));self.assertEqual(result['counts'],dict(zip('012345',[0,0,1,0,0,0])))
        self.assertEqual(result['objects'][0]['xmax'],40.)
        self.assertEqual(result['status'],'REVIEW');self.assertIsNone(result['predicted_label']);self.assertIsNone(result['prediction_score'])
        validate_result(result,'baseline3',frame.shape)
        args=self.engine.calls[0]
        for key,value in self.cfg['inference'].items():self.assertEqual(args[key],value)
    def test_empty_results_are_zero_counts_without_id0_object(self):
        self.engine.rows=[];result=self.make().infer(np.zeros((40,80,3),np.uint8),self.expected)
        self.assertEqual(result['objects'],[]);self.assertEqual(result['counts'],dict.fromkeys('012345',0))
        self.assertEqual(result['non_catalogue_count'],0)
    def test_id0_count_and_overlapping_cross_class_boxes_are_preserved(self):
        self.engine.rows=[[1.,1.,20.,20.,.9,0.],[1.,1.,20.,20.,.8,2.],[1.,1.,20.,20.,.7,3.]]
        result=self.make().infer(np.zeros((40,80,3),np.uint8),self.expected)
        self.assertEqual(result['non_catalogue_count'],1);self.assertEqual(len(result['objects']),3)
        self.assertEqual([o['detection_id'] for o in result['objects']],[0,1,2])
    def test_rejudge_does_not_detect_and_does_not_alias_objects(self):
        backend=self.make();result=backend.infer(np.zeros((40,80,3),np.uint8),self.expected)
        changed=backend.rejudge(result,dict.fromkeys('12345',0))
        self.assertEqual(len(self.engine.calls),1);self.assertEqual(changed['objects'],result['objects'])
        changed['objects'][0]['xmin']=99;self.assertEqual(result['objects'][0]['xmin'],10.)
        self.assertEqual(changed['status'],'REVIEW')
    def test_expected_counts_do_not_change_detection(self):
        backend=self.make();frame=np.zeros((40,80,3),np.uint8)
        a=backend.infer(frame,self.expected);b=backend.infer(frame,dict.fromkeys('12345',10))
        self.assertEqual(a['objects'],b['objects']);self.assertEqual(a['counts'],b['counts'])
    def test_invalid_frame_or_expected_rejected_before_detection(self):
        backend=self.make()
        for frame in [None,np.zeros((2,2),np.uint8),np.zeros((0,2,3),np.uint8),np.zeros((2,2,3),float)]:
            with self.subTest(frame_type=type(frame)),self.assertRaises(ValueError):backend.infer(frame,self.expected)
        for expected in [{},dict.fromkeys('012345',0),dict.fromkeys('12345',True),dict.fromkeys('12345',-1),dict.fromkeys('12345',1.)]:
            with self.subTest(expected=expected),self.assertRaises(ValueError):backend.infer(np.zeros((40,80,3),np.uint8),expected)
        self.assertEqual(self.engine.calls,[])
    def test_noncontiguous_input_supported(self):
        frame=np.zeros((40,160,3),np.uint8)[:,::2,:]
        result=self.make().infer(frame,self.expected);self.assertEqual(result['frame_size'],{'width':80,'height':40})
        self.assertTrue(self.engine.calls[0]['source'].flags.c_contiguous)
    def test_invalid_prediction_rejected_without_clipping_or_class_coercion(self):
        cases=[[0,0,0,1,.5,2],[-1,0,10,10,.5,2],[0,0,81,20,.5,2],[0,0,10,10,float('nan'),2],[0,0,float('inf'),10,.5,2],[0,0,10,10,.5,6],[0,0,10,10,.5,2.5],[0,0,10,10,1.1,2]]
        for row in cases:
            with self.subTest(row=row),self.assertRaises(ValueError):inference._observations([row],80,40)
    def test_tampered_rejudge_counts_rejected(self):
        backend=self.make();result=backend.infer(np.zeros((40,80,3),np.uint8),self.expected);result['counts']['2']=99
        with self.assertRaises(ValueError):backend.rejudge(result,self.expected)
    def test_boolean_cached_count_rejected(self):
        backend=self.make();result=backend.infer(np.zeros((40,80,3),np.uint8),self.expected);result['counts']['2']=True
        with self.assertRaises(ValueError):backend.rejudge(result,self.expected)
    def test_wrong_device_or_original_shape_rejected(self):
        backend=self.make();frame=np.zeros((40,80,3),np.uint8)
        for shape,is_cuda in [((20,40),True),((40,80),False)]:
            data=FakeData([]);data.is_cuda=is_cuda
            with patch.object(self.engine,'predict',return_value=[SimpleNamespace(orig_shape=shape,boxes=SimpleNamespace(data=data))]),self.assertRaises(ValueError):
                backend.infer(frame,self.expected)
    def test_version_mismatch_rejected_before_load(self):
        with patch.object(inference,'version',return_value='different'),patch.object(inference,'_load_model') as load,self.assertRaises(ValueError):
            inference.create_backend(model_id='baseline3',config={'config_path':str(self.path)})
        load.assert_not_called()
    def test_predict_failure_is_not_converted_to_empty_counts(self):
        backend=self.make()
        with patch.object(self.engine,'predict',side_effect=RuntimeError('injected GPU error')),self.assertRaisesRegex(RuntimeError,'GPU error'):
            backend.infer(np.zeros((40,80,3),np.uint8),self.expected)
    def test_sync_and_actual_p07_factory_loading(self):
        with patch.object(inference,'_cuda_runtime',return_value=self.runtime),patch.object(inference,'_load_model',return_value=self.engine):
            backend=load_backend('real','baseline3',{'bridge':'src.p06_inference:create_backend','model_config':{'config_path':str(self.path)}})
        self.assertFalse(backend.is_mock);backend.synchronize();self.assertEqual(self.sync,[0])
    def test_wrong_model_or_factory_options_rejected(self):
        for model,config in [('final',{}),('baseline1',{}),('baseline3',{'imgsz':960})]:
            with self.subTest(model=model,config=config),self.assertRaises(ValueError):inference.create_backend(model_id=model,config=config)
    def test_missing_weight_and_hash_mismatch(self):
        for remove in [False,True]:
            with self.subTest(remove=remove):
                if remove:self.weight.unlink()
                else:self.weight.write_bytes(b'changed')
                with self.assertRaises((ValueError,FileNotFoundError)):self.make()
    def test_frozen_class_or_inference_mismatch_rejected(self):
        for section,key,value in [('class_names','2','thin_screw'),('inference','imgsz',960),('inference','device',False)]:
            cfg=copy.deepcopy(self.cfg);cfg[section][key]=value;self.path.write_text(json.dumps(cfg),encoding='utf-8')
            with self.subTest(section=section,key=key),self.assertRaises(ValueError):self.make()
    def test_model_class_mismatch_rejected(self):
        self.engine.names={0:'person'}
        with self.assertRaises(ValueError):self.make()
    def test_cuda_unavailable_rejected(self):
        import torch
        with patch.object(torch.cuda,'is_available',return_value=False),self.assertRaisesRegex(RuntimeError,'CUDA'):
            inference._cuda_runtime()

if __name__=='__main__':unittest.main()
