"""Dataset builder regressions: synthetic fixtures, no real training or evaluation."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from src.p06_data import DataError
from tools.p06_build_dataset import create_plan, validate_plan, build_dataset, validate_artifacts


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name); self.source=self.root/'source'; self.source.mkdir()
        for index,name in enumerate(('pass01','fail11','fail13')):
            im=Image.new('RGB',(120,80),(20+index,50,90))
            exif=Image.Exif(); exif[274]=6
            im.save(self.source/(name+'.jpg'),exif=exif)
            objects=[] if name=='fail13' else [dict(label='non_catalogue' if name=='fail11' else 'long_screw',
                shape_type='rectangle',points=[[10,20],[40,70]],flags={})]
            doc=dict(imagePath=name+'.jpg',imageWidth=80,imageHeight=120,flags={},shapes=objects)
            (self.source/(name+'.json')).write_text(json.dumps(doc))

    def plan(self):
        return create_plan(self.source,{'fail13.jpg'},{'fail13.jpg'})

    def test_fixed_plan_and_provisional_scene_provenance(self):
        a=self.plan(); b=self.plan()
        self.assertEqual(a,b)
        self.assertEqual(len(a['records']),3)
        self.assertTrue(all(r['scene_id'].startswith('P06_TEMP_') for r in a['records']))
        self.assertTrue(all(r['scene_authority']=='P06_provisional_pending_P03_P08' for r in a['records']))
        self.assertEqual(a['summary']['train']['classes'],[1,1,0,0,0,0])

    def test_bad_partitions_and_id0_val_rejected(self):
        plan=self.plan(); plan['records'][0]['split']='test'
        with self.assertRaises(DataError): validate_plan(plan)
        plan=self.plan()
        for row in plan['records']:
            if row['image_id']=='fail11.jpg': row['split']='val'
        with self.assertRaisesRegex(DataError,'ID 0'): validate_plan(plan)
        plan=self.plan(); plan['records'][1]['scene_id']=plan['records'][0]['scene_id']
        plan['records'][1]['split']='val'
        with self.assertRaises(DataError): validate_plan(plan)

    def test_missing_json_and_unknown_validation_name_rejected(self):
        with self.assertRaises(DataError): create_plan(self.source,{'absent.jpg'},{'fail13.jpg'})
        (self.source/'pass01.json').unlink()
        with self.assertRaises(DataError): self.plan()

    def test_build_corrected_png_txt_empty_and_roundtrip(self):
        plan=self.plan(); out=self.root/'dataset'
        report=build_dataset(plan,out,official_check=False)
        self.assertEqual(report['instances'],2)
        self.assertLess(report['max_roundtrip_error_pixels'],1e-6)
        self.assertEqual((out/'labels/val/fail13.txt').read_bytes(),b'')
        self.assertEqual((out/'labels/train/fail11.txt').read_text().split()[0],'0')
        with Image.open(out/'images/train/pass01.png') as im:
            self.assertEqual(im.size,(80,120)); self.assertEqual(im.getexif().get(274,1),1)
        validate_artifacts(plan,out)
        with self.assertRaises(FileExistsError): build_dataset(plan,out,official_check=False)

    def test_source_change_and_txt_corruption_rejected(self):
        plan=self.plan()
        (self.source/'pass01.json').write_text('{}')
        with self.assertRaises(DataError): build_dataset(plan,self.root/'bad',official_check=False)
        self.assertFalse((self.root/'bad').exists())

    def test_post_build_corrupt_label_rejected(self):
        plan=self.plan(); out=self.root/'dataset'; build_dataset(plan,out,official_check=False)
        (out/'labels/train/pass01.txt').write_text('1 0.5 0.5 -0.2 0.4\n')
        with self.assertRaises(DataError): validate_artifacts(plan,out)

    def test_failed_official_check_never_publishes(self):
        plan=self.plan(); out=self.root/'dataset'
        with patch('tools.p06_build_dataset.validate_ultralytics',side_effect=RuntimeError('injected loader failure')):
            with self.assertRaises(RuntimeError): build_dataset(plan,out)
        self.assertFalse(out.exists())
        self.assertEqual([p.name for p in self.root.iterdir()],['source'])


if __name__=='__main__': unittest.main()
