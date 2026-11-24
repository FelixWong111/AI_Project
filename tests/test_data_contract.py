"""Guard source isolation and reject ambiguous/contradictory development labels."""
import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image

import src.data.development as data
from src.inference.batch import load_cases


class DataContractTests(unittest.TestCase):
    def test_invalid_counts_are_not_coerced(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'cases.json'
            for value in (True,1.5,-1):
                p.write_text(json.dumps([dict(image_id='a.jpg',expected_counts=dict(zip('12345',[value,1,1,1,1])))]))
                with self.assertRaises(ValueError):load_cases(p)

    def test_review_and_test_never_opened_and_scene_confirmation_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);part=root/'development';(part/'images/train').mkdir(parents=True);(part/'images/validation').mkdir()
            rows=[];objects=[]
            for index,(split,label) in enumerate([('train','PASS'),('train','FAIL'),('validation','PASS'),('validation','FAIL')]):
                iid=f'{index}.png';Image.new('RGB',(20,20),(index*35,100,0)).save(part/'images'/split/iid)
                counts=[1]*5 if label=='PASS' else [0]*5
                rows.append(dict(image_id=iid,scene_id='TBD',split=split,width=20,height=20,annotation_status='reviewed',
                                 **dict(zip(data.COUNT_FIELDS,counts)),non_catalogue_count=0,reference_label_all_one=label))
                if label=='PASS':
                    objects.extend(dict(image_id=iid,object_id=str(c),class_id=c,xmin=0,ymin=0,xmax=10,ymax=10,ignore=0) for c in range(1,6))
            rows.extend([dict(rows[0],image_id='must-not-open-test.png',split='test'),
                         dict(rows[0],image_id='must-not-open-review.png',annotation_status='REVIEW')])
            for name,values in [('ground_truth_images.csv',rows),('ground_truth_objects.csv',objects)]:
                with (part/name).open('w',newline='',encoding='utf-8-sig') as stream:
                    writer=csv.DictWriter(stream,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
            settings=dict(expected_counts=dict.fromkeys('12345',1),reference_label_field='reference_label_all_one')
            with patch.object(data,'ROOT',root):
                with self.assertRaisesRegex(ValueError,'Scene identity'):data.prepare(part,settings)
                settings['scene_confirmation']=dict(each_image_is_distinct_scene=True,source='fixture user confirmation')
                records,evidence=data.prepare(part,settings)
                self.assertEqual(len(records),4)
                self.assertEqual(evidence['exclusions'][0]['reason'],'REVIEW')
                self.assertEqual(evidence['summary']['validation']['scenes'],2)
                # A wrong stored label must never be silently regenerated.
                p=part/'ground_truth_images.csv';s=p.read_text(encoding='utf-8-sig');p.write_text(s.replace('PASS','FAIL'),encoding='utf-8-sig')
                with self.assertRaisesRegex(ValueError,'Label/count conflict'):data.prepare(part,settings)
