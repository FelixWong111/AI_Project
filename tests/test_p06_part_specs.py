"""Supplier reference integrity and descriptive comparison; synthetic inputs only."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tools.p06_check_part_specs import validate_catalogue, compare_record


class PartSpecsTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.package=Path(self.temp.name)
        self.config=json.loads(Path('config/p06_part_specs_candidate.json').read_text(encoding='utf-8'))
        raw={'class_zero':{'class_id':0,'class_name':'non_catalogue','dimensions':None},'items':[]}
        for item in self.config['items']:
            path=self.package/item['source_image'];path.parent.mkdir(exist_ok=True)
            path.write_bytes(b'synthetic-reference-'+str(item['class_id']).encode())
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            item['source_image_sha256']=digest
            source=dict(class_id=item['class_id'],class_name=item['class_name'],
                        nominal_spec=item['supplier_spec'],source_image_in_package=item['source_image'],source_image_sha256=digest)
            source.update({d['supplier_field']:d['reference_mm'] for d in item['dimensions'].values()})
            source.update(item['reference_only_dimensions'])
            raw['items'].append(source)
        doc=self.package/'p06_supplier_specs.json';doc.write_text(json.dumps(raw),encoding='utf-8')
        self.config['source']['supplier_json_sha256']=hashlib.sha256(doc.read_bytes()).hexdigest()

    def test_mapping_and_all_nine_user_provisional_tolerances(self):
        validate_catalogue(self.config,self.package)
        expected={1:{'underhead_length':(25,1),'thread_major_diameter':(4,.3)},
                  2:{'underhead_length':(10,1),'thread_major_diameter':(5,.3)},
                  3:{'underhead_length':(10,1),'thread_major_diameter':(4,.3)},
                  4:{'across_flats':(7,.5)},5:{'outer_diameter':(15,.75),'inner_diameter':(5,.5)}}
        for item in self.config['items']:
            self.assertEqual({k:(v['reference_mm'],v['provisional_allowance_mm']) for k,v in item['dimensions'].items()},expected[item['class_id']])
            self.assertIsNone(item['iso_din_standard'])
        self.assertFalse(self.config['formal_decision_enabled'])

    def test_image_tampering_missing_and_unsafe_path_rejected(self):
        path=self.package/self.config['items'][0]['source_image'];path.write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError,'SHA256'):validate_catalogue(self.config,self.package)
        path.unlink()
        with self.assertRaisesRegex(ValueError,'Missing'):validate_catalogue(self.config,self.package)
        self.config['items'][0]['source_image']='../escape.jpg'
        with self.assertRaises(ValueError):validate_catalogue(self.config,self.package)

    def test_unknown_mapping_and_invented_unsupported_dimension_rejected(self):
        cfg=copy.deepcopy(self.config);cfg['items'][0]['class_id']=2
        with self.assertRaises(ValueError):validate_catalogue(cfg,self.package)
        cfg=copy.deepcopy(self.config);cfg['items'][4]['dimensions']['thickness']={}
        with self.assertRaises(ValueError):validate_catalogue(cfg,self.package)

    def test_reference_value_mismatch_and_invalid_allowance_rejected(self):
        cfg=copy.deepcopy(self.config);cfg['items'][4]['dimensions']['inner_diameter']['reference_mm']=5.3
        with self.assertRaisesRegex(ValueError,'Supplier value'):validate_catalogue(cfg,self.package)
        for value in (-.1,float('nan'),True):
            cfg=copy.deepcopy(self.config);cfg['items'][0]['dimensions']['underhead_length']['provisional_allowance_mm']=value
            with self.assertRaises(ValueError):validate_catalogue(cfg,self.package)

    def test_difference_is_not_measurement_error_or_product_decision(self):
        row=dict(class_id=2,image_id='synthetic.jpg',object_id='x',status='MEASURED_CANDIDATE',reason='',
                 dimensions={'underhead_length':{'mm_estimate':9.5},'thread_major_diameter':{'mm_estimate':7.12}})
        result=compare_record(row,self.config)
        self.assertAlmostEqual(result[0]['difference_from_supplier_reference_mm'],-.5)
        self.assertAlmostEqual(result[1]['difference_from_supplier_reference_mm'],2.12)
        for r in result:
            self.assertEqual(r['decision'],'REVIEW')
            self.assertNotIn('measurement_error_mm',r)
            self.assertNotIn('within_tolerance',r)

    def test_failed_measurement_has_no_zero_substitution(self):
        row=dict(class_id=5,image_id='synthetic.jpg',status='UNMEASURABLE',reason='INNER_HOLE_NOT_UNAMBIGUOUS',dimensions={})
        result=compare_record(row,self.config)
        self.assertEqual(len(result),2)
        self.assertTrue(all(r['mm_estimate'] is None and r['difference_from_supplier_reference_mm'] is None for r in result))

    def test_missing_nonfinite_and_class_zero_are_explicit(self):
        row=dict(class_id=1,image_id='synthetic.jpg',status='MEASURED_CANDIDATE',reason='',dimensions={})
        self.assertTrue(all(r['reason']=='MISSING_DIMENSION' for r in compare_record(row,self.config)))
        row['dimensions']={'underhead_length':{'mm_estimate':float('nan')}}
        with self.assertRaises(ValueError):compare_record(row,self.config)
        row['class_id']=0
        self.assertEqual(compare_record(row,self.config),[])


if __name__=='__main__':unittest.main()
