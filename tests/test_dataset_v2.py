"""Data conversion contract and held-out access guards; no training or model evaluation."""
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from src.data.dataset_v2 import (parse_annotation, count_label, prepare_development,
                                 approved_development_path, save_json, ROOT)
from collections import Counter


class DataV2Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.base=Path(self.tmp.name);self.json=self.base/'sample.json'
        self.info=dict(width=100,height=80,exif_orientation=1)

    def tearDown(self):self.tmp.cleanup()

    def document(self,shapes):
        save_json(self.json,dict(imagePath='sample.jpg',imageWidth=100,imageHeight=80,shapes=shapes,flags={}))

    def shape(self,label='non_catalogue',points=None,kind='rectangle'):
        return dict(label=label,points=points or [[10,20],[30,50]],shape_type=kind,flags={},group_id=None)

    def test_polygon_is_tight_aabb_with_official_zero(self):
        self.document([self.shape(points=[[10,20],[30,21],[25,50]],kind='polygon')])
        objects,issues,_=parse_annotation(self.json,self.info)
        self.assertFalse(issues);self.assertEqual(objects[0]['class_id'],0)
        self.assertEqual(objects[0]['xyxy'],[10,20,30,50])

    def test_missing_json_is_not_empty_scene(self):
        objects,issues,_=parse_annotation(self.json,self.info)
        self.assertEqual(objects,[]);self.assertIn('MISSING_JSON',issues)

    def test_invalid_class_and_outside_box_are_reported(self):
        for shape in [self.shape(label='M4'),self.shape(points=[[-1,10],[30,40]]),self.shape(points=[[10,10],[10,20]])]:
            self.document([shape]);objects,issues,_=parse_annotation(self.json,self.info)
            self.assertEqual(objects,[]);self.assertTrue(issues)

    def test_duplicate_box_not_double_counted(self):
        self.document([self.shape(),self.shape()]);objects,issues,_=parse_annotation(self.json,self.info)
        self.assertEqual(len(objects),1);self.assertIn('DUPLICATE_BOX',issues[0])

    def test_label_uses_counts_and_zero_not_filename(self):
        counts=Counter({1:1,2:1,3:1,4:1,5:1});self.assertEqual(count_label(counts),'PASS')
        counts[0]=1;self.assertEqual(count_label(counts),'FAIL')
        self.assertEqual(count_label(Counter()),'FAIL')

    def test_candidate_cannot_enter_formal_data_loader(self):
        save_json(self.base/'state.json',dict(status='CANDIDATE_NOT_READY'))
        # A relative project path is required by the real marker; use a project-local fixture.
        scratch=ROOT/'tmp/data_v2_tests';scratch.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as folder:
            base=Path(folder);state=base/'state.json';save_json(state,dict(status='CANDIDATE_NOT_READY'))
            save_json(base/'dataset_v2.json',dict(dataset_root=base.relative_to(ROOT).as_posix(),status_file=state.relative_to(ROOT).as_posix()))
            with self.assertRaisesRegex(ValueError,'CANDIDATE_NOT_READY'):prepare_development(base,{})

    def test_legacy_path_guard_still_excludes_test(self):
        self.assertTrue(approved_development_path(self.base,dict(split='train'),self.base/'images/train/a.jpg'))
        self.assertFalse(approved_development_path(self.base,dict(split='train'),self.base/'images/test/a.jpg'))
        self.assertFalse(approved_development_path(self.base,dict(split='test'),self.base/'images/test/a.jpg'))

    def test_p08_checks_only_explicit_stage_and_blocks_unfrozen_evaluation(self):
        from p08_run import evaluation_inputs
        from src.data.dataset_v2 import save_csv
        root=self.base;data=root/'data/fixture';data.mkdir(parents=True)
        save_json(data/'status.json',dict(status='CANDIDATE_NOT_READY',frozen=False))
        save_json(data/'counts.json',[dict(image_id='one.jpg',expected_counts={str(c):1 for c in range(1,6)})])
        save_csv(data/'images.csv',[dict(image_id='one.jpg',evaluation_stage='stage2',width=100,height=80)])
        save_csv(data/'objects.csv',[],['image_id','class_id','xmin','ymin','xmax','ymax'])
        (data/'one.jpg').write_bytes(b'fixture placeholder; check-data-only does not decode')
        (data/'stage.txt').write_text('one.jpg\n')
        config=dict(schema='inspection-evaluation/v2',evaluation_stage='stage2',dataset='data/fixture',input_dir='data/fixture',
                    ground_truth_images='data/fixture/images.csv',ground_truth_objects='data/fixture/objects.csv',counts_file='data/fixture/counts.json',
                    split_list='data/fixture/stage.txt',dataset_status='data/fixture/status.json')
        save_json(data/'config.json',config)
        with patch('p08_run.ROOT',root):
            _,result=evaluation_inputs(data/'config.json',check_only=True)
            self.assertEqual(result['images'],1);self.assertEqual(result['metrics'],'NOT_RUN')
            with self.assertRaisesRegex(ValueError,'not READY'):evaluation_inputs(data/'config.json')
            config['evaluation_stage']='stage3';save_json(data/'config.json',config)
            with self.assertRaisesRegex(ValueError,'Stage list'):evaluation_inputs(data/'config.json',check_only=True)


if __name__=='__main__':unittest.main()
