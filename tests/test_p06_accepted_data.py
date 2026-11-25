"""New P03 workflow: synthetic accepted package, no real Test access."""
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from PIL import Image
from src.data.p06_labelme import CLASSES,DataError,from_yolo,to_yolo
from tools.p06_accepted_data import accepted_plan,prepare_runtime,check_sources,parse_txt


class AcceptedTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name); self.source=self.root/'source'; self.audit=self.root/'audit'; self.audit.mkdir()
        for split in ('train','validation'):
            (self.source/'images'/split).mkdir(parents=True); (self.source/'labels'/split).mkdir(parents=True)
        records=[]; image_rows=[]; object_rows=[]
        for i,(name,split,cid) in enumerate((('a.jpg','train',0),('b.jpg','validation',1),('empty.jpg','train',None))):
            path=self.source/'images'/split/name
            Image.new('RGB',(200,100),(40+60*i,20,60)).save(path)
            objects=[] if cid is None else [dict(label=CLASSES[cid],shape_type='rectangle',points=[[20,10],[100,60]],flags={})]
            path.with_suffix('.json').write_text(json.dumps(dict(imagePath=name,imageWidth=200,imageHeight=100,shapes=objects,flags={})))
            (self.source/'labels'/split/Path(name).with_suffix('.txt')).write_text('' if cid is None else f'{cid} 0.3000000000 0.3500000000 0.4000000000 0.5000000000\n')
            counts=[int(cid==c) for c in range(6)]
            records.append(dict(image_id=name,scene_id='user_scene_'+name,split='val' if split=='validation' else 'train',source_split=split,
                scene_authority='P06_provisional_user_confirmed_independent_rearrangement',scene_basis='user confirmed',sha256=hashlib.sha256(path.read_bytes()).hexdigest(),pixel_sha256='audited'))
            image_rows.append(dict(image_id=name,scene_id='TBD',split=split,width=200,height=100,annotation_status='reviewed',
                **dict(zip(['non_catalogue_count',*[n+'_count' for n in CLASSES[1:]]],counts))))
            if cid is not None: object_rows.append(dict(image_id=name,object_id=name+'_1',class_id=cid,xmin=20,ymin=10,xmax=100,ymax=60,ignore=0))
        self.write_csv(self.audit/'p06_derived_manifest_reviewed.csv',records)
        self.write_csv(self.source/'ground_truth_images.csv',image_rows)
        self.write_csv(self.source/'ground_truth_objects.csv',object_rows)
        self.source.joinpath('classes.txt').write_text('\n'.join(CLASSES)+'\n')
        self.source.joinpath('data.yaml').write_text('train: images/train\nval: images/validation\nnames:\n'+''.join(f'  {c}: {n}\n' for c,n in enumerate(CLASSES)))
        (self.audit/'acceptance_reviewed.json').write_text(json.dumps(dict(status='DATA_PASS_WITH_TOOL_COMPATIBILITY_WARNING',source=str(self.source),valid_images=3,objects=2,
            splits=dict(train=dict(images=2,classes=[1,0,0,0,0,0],empty=['empty.jpg']),val=dict(images=1,classes=[0,1,0,0,0,0],empty=[])))))
        self.freeze()

    def write_csv(self,path,rows):
        with path.open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

    def freeze(self):
        hashes={p.relative_to(self.source).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in self.source.rglob('*') if p.is_file()}
        (self.audit/'source_sha256.json').write_text(json.dumps(hashes))

    def plan(self): return accepted_plan(self.source,self.audit)

    def test_precision_aware_boundary_preserves_strict_source_rules(self):
        values=[.8646384480,.2762324959,.2707231041,.1526634045]
        with self.assertRaises(DataError): from_yolo(values,3024,4032)
        box=from_yolo(values,3024,4032,decimal_places=10)
        self.assertEqual(box[2],3024.)
        with self.assertRaises(DataError): from_yolo([.86464,.27623,.27073,.15266],3024,4032,decimal_places=10)
        with self.assertRaises(DataError): to_yolo([1,1,3024.0000001,30],3024,4032)
        with self.assertRaises(ValueError): parse_txt('0 nan .5 .2 .2',3024,4032)

    def test_accepted_plan_and_runtime_no_source_cache_or_reencoding(self):
        plan=self.plan(); runtime=self.root/'runtime'
        prepare_runtime(plan,runtime,official_check=False)
        self.assertTrue((runtime/'images/val/b.jpg').samefile(self.source/'images/validation/b.jpg'))
        self.assertEqual((runtime/'labels/train/empty.txt').read_bytes(),b'')
        self.assertFalse(list(self.source.rglob('*.cache')))
        check_sources(plan)
        with self.assertRaises(ValueError): prepare_runtime(plan,runtime,official_check=True)

    def test_changed_source_and_test_yaml_rejected(self):
        (self.source/'data.yaml').write_text('test: must-not-open\n')
        with self.assertRaises(ValueError): self.plan()
        self.freeze()
        with self.assertRaises(ValueError): self.plan()

    def test_changed_manifest_and_bad_scene_rejected(self):
        p=self.audit/'p06_derived_manifest_reviewed.csv'
        with p.open() as f: rows=list(csv.DictReader(f))
        rows[0]['split']='test';self.write_csv(p,rows)
        with self.assertRaises(ValueError): self.plan()

    def test_runtime_tamper_and_source_change_fail(self):
        plan=self.plan(); runtime=self.root/'runtime';prepare_runtime(plan,runtime,official_check=False)
        (runtime/'labels/val/b.txt').write_text('changed')
        with self.assertRaises(ValueError): prepare_runtime(plan,runtime,official_check=False)
        (self.source/'ground_truth_objects.csv').write_text('changed')
        with self.assertRaises(ValueError): check_sources(plan)


if __name__=='__main__':unittest.main()
