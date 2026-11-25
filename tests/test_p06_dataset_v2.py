"""Synthetic v2 manifest/EXIF/atomic-publish regression checks."""
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
from src.data.p06_labelme import DataError
from tools.p06_build_dataset import create_plan, build_dataset, validate_artifacts, validate_plan
from tools.p06_build_dataset_v2 import create_v2_plan, validate_v2_plan


class V2Tests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name); self.source=self.root/'source'; self.source.mkdir()
        for i,name in enumerate(('pass01','fail11','fail13')):
            exif=Image.Exif(); exif[274]=6
            Image.new('RGB',(120,80),(20+60*i,50,90)).save(self.source/(name+'.jpg'),exif=exif)
            objects=[] if name=='fail13' else [dict(label='non_catalogue' if name=='fail11' else 'long_screw',
                shape_type='rectangle',points=[[10,20],[40,70]],flags={})]
            (self.source/(name+'.json')).write_text(json.dumps(dict(imagePath=name+'.jpg',imageWidth=80,imageHeight=120,flags={},shapes=objects)))
        original=create_plan(self.source,{'fail13.jpg'},{'fail13.jpg'})
        self.old=copy.deepcopy(original); self.old['records']=[r for r in original['records'] if r['image_id']!='fail11.jpg']
        self.rows=[]
        for row in original['records']:
            self.rows.append(dict(image_id=row['image_id'],scene_id=row['scene_id'],
                split='val' if row['image_id']=='fail11.jpg' else row['split'],
                scene_authority='P06_provisional',scene_basis='user confirmed rearrangement',reason='approved fixture',
                provenance='new275' if row['image_id']=='fail11.jpg' else 'old51',
                sha256=original['source_sha256'][row['image_id']],pixel_sha256=row['display_pixel_sha256'],
                **{f'count_{c}':row['counts'][c] for c in range(6)}))
        self.manifest=self.root/'manifest.csv'; self.write_manifest()

    def write_manifest(self):
        with self.manifest.open('w',newline='',encoding='utf-8-sig') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(self.rows[0])); writer.writeheader(); writer.writerows(self.rows)

    def plan(self):
        return create_v2_plan(self.source,self.manifest,self.old,{'fail13.jpg'})

    def test_v2_allows_id0_val_preserves_old_policy_and_roundtrip(self):
        plan=self.plan(); validate_v2_plan(plan)
        with self.assertRaises(DataError): validate_plan(plan)
        out=self.root/'dataset'; result=build_dataset(plan,out,official_check=False,version=2)
        self.assertEqual(result['split_classes']['val'][0],1)
        self.assertEqual((out/'labels/val/fail13.txt').read_bytes(),b'')
        with Image.open(out/'images/val/fail11.png') as im:
            self.assertEqual(im.size,(80,120)); self.assertEqual(im.getexif().get(274,1),1)
        self.assertLess(validate_artifacts(plan,out)['max_roundtrip_error_pixels'],1e-6)
        self.assertNotIn('Validation 没有 ID 0',(out/'dataset_report.md').read_text(encoding='utf-8'))
        with self.assertRaises(FileExistsError): build_dataset(plan,out,official_check=False,version=2)

    def test_manifest_test_duplicate_and_missing_rejected(self):
        for mode in ('test','duplicate','missing'):
            saved=copy.deepcopy(self.rows)
            if mode=='test': self.rows[0]['split']='test'
            elif mode=='duplicate': self.rows.append(copy.deepcopy(self.rows[0]))
            else: self.rows.pop()
            self.write_manifest()
            with self.assertRaises((DataError,ValueError)): self.plan()
            self.rows=saved

    def test_old_validation_cannot_move(self):
        next(r for r in self.rows if r['image_id']=='fail13.jpg')['split']='train'
        self.write_manifest()
        with self.assertRaises(DataError): self.plan()

    def test_manifest_counts_hash_and_scene_rejected(self):
        for field,value in (('count_0',9),('sha256','wrong'),('scene_id','')):
            saved=copy.deepcopy(self.rows); self.rows[0][field]=value; self.write_manifest()
            with self.assertRaises(DataError): self.plan()
            self.rows=saved

    def test_scene_leakage_and_post_plan_split_tamper(self):
        plan=self.plan()
        bad=copy.deepcopy(plan); bad['records'][0]['split']='train'
        with self.assertRaises(DataError): validate_v2_plan(bad)
        self.rows[0]['scene_id']=next(r['scene_id'] for r in self.rows if r['split']=='train')
        self.write_manifest()
        with self.assertRaises(DataError): self.plan()

    def test_source_change_and_failed_loader_never_publish(self):
        plan=self.plan(); out=self.root/'dataset'
        with patch('tools.p06_build_dataset.validate_ultralytics',side_effect=RuntimeError('injected')):
            with self.assertRaises(RuntimeError): build_dataset(plan,out,version=2)
        self.assertFalse(out.exists())
        (self.source/'fail11.json').write_text('{}')
        with self.assertRaises(DataError): build_dataset(plan,out,version=2,official_check=False)
        self.assertFalse(out.exists())

    def test_approved_manifest_file_change_after_plan_rejected(self):
        plan=self.plan()
        self.manifest.write_text('changed after approval',encoding='utf-8')
        with self.assertRaises(DataError): validate_v2_plan(plan)


if __name__=='__main__': unittest.main()
