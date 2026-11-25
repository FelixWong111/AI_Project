# -*- coding: utf-8 -*-
"""UTF-8 report regression, using synthetic JSON/CSV only."""
import copy
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tools.p06_report_geometry_stage1 import build_report, write_report, validate_report_text


class Stage1ReportTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.run=self.root/'train_full_final';self.run.mkdir()
        def result(status,dimensions,reason=''):
            return dict(status=status,dimensions=dimensions,reason=reason,diagnostics={},decision=None)
        self.records=[dict(image_id='训练图.jpg',object_id='合成对象',class_id=5,split='train',
            before=result('UNMEASURABLE',{},'INNER_HOLE_NOT_UNAMBIGUOUS'),
            after=result('MEASURED_CANDIDATE',{'outer_diameter':{'pixels':300.,'mm_estimate':15.},
                                                'inner_diameter':{'pixels':100.,'mm_estimate':5.}}))]
        self.summary={str(i):dict(objects=int(i==5),before=dict(candidate_count=0,failure_reasons={'INNER_HOLE_NOT_UNAMBIGUOUS':1} if i==5 else {}),
            after=dict(candidate_count=int(i==5),failure_reasons={}),newly_refused=0,newly_available=int(i==5),after_partial_review_only=0) for i in (1,2,3,5)}
        self.save(self.run/'records.json',self.records);self.save(self.run/'summary.json',self.summary)
        self.save(self.run/'source_image_sha256.json',{'训练图.jpg':'synthetic-hash-not-a-real-image'})
        self.save(self.root/'selection.json',dict(train_image_selection=['训练图.jpg'],scale_px_per_mm=[20,20]))
        with (self.run/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['image_id','object_id','class_id','phase','status','reason','dimension','pixels','mm_estimate','line_xy']);w.writeheader()
            for phase in ('before','after'):
                r=self.records[0][phase]
                for name,d in (r['dimensions'].items() or [('none',{})]):
                    w.writerow(dict(image_id='训练图.jpg',object_id='合成对象',class_id=5,phase=phase,status=r['status'],reason=r['reason'],
                                    dimension=name,pixels=d.get('pixels',''),mm_estimate=d.get('mm_estimate',''),line_xy='[]'))

    def save(self,path,value):path.write_text(json.dumps(value,ensure_ascii=False),encoding='utf-8')

    def test_reads_json_and_bom_csv_without_any_image_access(self):
        text=build_report(self.root)
        self.assertIn('阶段 1',text);self.assertIn('测量',text)
        self.assertIn('| 5 flat_washer | 1 | 0 | 1 | 0 | 1 | 0 |',text)
        self.assertIn('训练图.jpg',text);self.assertIn('合成对象',text)
        self.assertNotIn('?',text);self.assertNotIn('\ufffd',text)
        self.assertNotIn('|\n\n|',text)

    def test_replaces_corruption_from_evidence_not_from_old_markdown(self):
        damaged=self.root/'report.md';damaged.write_bytes(b'??? old numbers 999999')
        write_report(self.root,replace=True)
        text=damaged.read_bytes().decode('utf-8',errors='strict')
        self.assertIn('阶段 1',text);self.assertNotIn('999999',text);self.assertNotIn('???',text)
        self.assertIn('原始记录',text)

    def test_existing_report_requires_explicit_replace(self):
        path=self.root/'report.md';path.write_bytes(b'keep')
        with self.assertRaises(FileExistsError):write_report(self.root)
        self.assertEqual(path.read_bytes(),b'keep')

    def test_wrong_summary_is_refused_without_overwriting(self):
        broken=copy.deepcopy(self.summary);broken['5']['after']['candidate_count']=99
        self.save(self.run/'summary.json',broken)
        path=self.root/'report.md';path.write_bytes(b'keep')
        with self.assertRaisesRegex(ValueError,'SUMMARY_MISMATCH'):write_report(self.root,replace=True)
        self.assertEqual(path.read_bytes(),b'keep')

    def test_wrong_csv_value_is_refused(self):
        path=self.run/'comparison.csv';s=path.read_text(encoding='utf-8-sig');path.write_text(s.replace('300.0,15.0','300.0,99.0'),encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'CSV_MISMATCH'):build_report(self.root)

    def test_lossy_template_is_detected(self):
        for text in ('# ???\n\n????????', '# 阶段 1\n测量\ufffd候选', '# 阶段 1\n测量候选????'):
            with self.assertRaises(ValueError):validate_report_text(text)

    def test_ascii_console_and_disabled_utf8_mode_do_not_damage_report(self):
        script=Path(__file__).resolve().parents[1]/'tools/p06_report_geometry_stage1.py'
        env=os.environ.copy();env.update(PYTHONUTF8='0',PYTHONIOENCODING='ascii:replace')
        p=subprocess.run([sys.executable,'-B','-X','utf8=0',str(script),'--experiment',str(self.root)],
                         env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        self.assertEqual(p.returncode,0,p.stderr.decode('ascii',errors='replace'))
        raw=(self.root/'report.md').read_bytes()
        self.assertIn('阶段 1'.encode('utf-8'),raw);self.assertIn('训练图.jpg'.encode('utf-8'),raw)
        self.assertNotIn(b'??',raw)

    def test_non_train_records_are_refused(self):
        self.records[0]['split']='validation';self.save(self.run/'records.json',self.records)
        with self.assertRaisesRegex(ValueError,'TRAIN_ONLY'):build_report(self.root)


if __name__=='__main__':unittest.main()
