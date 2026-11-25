"""Resolution forwarding and comparison guards, using synthetic metrics only."""
import copy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from tools import p06_compare_resolution as resolution

from tools.p06_train_baseline import validate_best
from tools.p06_compare_resolution import changes,summarize

class ResolutionTests(unittest.TestCase):
    def test_existing_output_refused_before_environment_and_data_checks(self):
        with TemporaryDirectory() as tmp,patch.object(resolution,'OUT',Path(tmp)),patch.object(resolution,'inspect_environment') as environment,patch('sys.argv',['compare']):
            with self.assertRaisesRegex(ValueError,'overwrite'): resolution.main()
            environment.assert_not_called()
            self.assertEqual(list(Path(tmp).iterdir()),[])

    def test_validation_forwards_only_requested_resolution(self):
        calls=[];callbacks={}
        class Matrix:
            def astype(self,kind): return self
            def tolist(self): return [[0]*7 for _ in range(7)]
        metrics=SimpleNamespace(nt_per_class=[0,1,0,0,0,0],
            box=SimpleNamespace(ap_class_index=[1],mean_results=lambda:[.8,.9,.7,.6],class_result=lambda i:[.8,.9,.7,.6]),
            confusion_matrix=SimpleNamespace(matrix=Matrix()))
        def val(**kwargs):
            calls.append(kwargs)
            for fn in callbacks.get('on_val_start',[]): fn(SimpleNamespace(device=SimpleNamespace(type='cuda')))
            for fn in callbacks.get('on_val_end',[]): fn(SimpleNamespace(seen=1))
            return metrics
        def add(event,fn): callbacks.setdefault(event,[]).append(fn)
        model=SimpleNamespace(val=val,add_callback=add)
        with TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'best_validation').mkdir()
            for size in [640,960]: validate_best(model,root,root,dict(summary=dict(val=dict(images=1,classes=[0,1,0,0,0,0]))),imgsz=size)
        self.assertEqual(calls[0]['imgsz'],640);self.assertEqual(calls[1]['imgsz'],960)
        self.assertEqual({k:v for k,v in calls[0].items() if k!='imgsz'},{k:v for k,v in calls[1].items() if k!='imgsz'})

    def test_comparison_rejects_missing_or_reordered_images(self):
        with self.assertRaisesRegex(ValueError,'membership'): changes([dict(image_id='a')],[])
        with self.assertRaisesRegex(ValueError,'membership'): changes([dict(image_id='a')],[dict(image_id='b')])

    def test_diagnostics_keep_class_confusion_and_extra_prediction_separate(self):
        a=dict(image_id='synthetic.jpg',detections=2,class_counts=[0,0,1,1,0,0],confusions_gt_to_pred=[[3,2]],unmatched_gt=[],unmatched_predictions=[dict(class_id=3,possible_duplicate=True)])
        b=copy.deepcopy(a);b.update(detections=1,class_counts=[0,0,0,1,0,0],confusions_gt_to_pred=[],unmatched_predictions=[])
        self.assertEqual(summarize([a])['class_confusions'],1)
        self.assertEqual(summarize([a])['possible_duplicates'],1)
        self.assertEqual(changes([a],[b])[0]['960']['unmatched_predictions'],[])
        self.assertEqual(changes([b],[b]),[])

if __name__=='__main__': unittest.main()
