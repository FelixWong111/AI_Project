"""Synthetic stage-one functional checks, never model or metrology accuracy."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import cv2

from src.p06_size_evidence import compare_dimensions, SizeEvidenceProcessor, validate_specs, validate_calibration
from src.p06_inference import _observations


class SizeEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.specs=json.loads((Path(__file__).resolve().parents[1]/'config/p06_part_specs_candidate.json').read_text(encoding='utf-8'))

    def measured(self, **dimensions):
        return dict(class_id=2,status='PARTIAL_CANDIDATE',
                    dimensions={k:dict(mm_estimate=v) for k,v in dimensions.items()})

    def compare(self,measurement, error=0, **options):
        quality={k:dict(verified=True,max_error_mm=error,evidence_id='synthetic-fixture')
                 for k in measurement['dimensions']}
        return compare_dimensions(measurement,self.specs,quality,calibration_validated=True,
                                  family_confirmed=options.get('family_confirmed',True))

    def test_one_reliable_diameter_excludes_all_screw_sizes(self):
        result=self.compare(self.measured(thread_major_diameter=6),.1)
        self.assertEqual(result['conclusion'],'OUTSIDE_FAMILY_CANDIDATE')
        self.assertEqual(result['compatible_class_ids'],[])
        self.assertEqual(result['status'],'REVIEW')

    def test_excluding_predicted_class_preserves_other_legal_classes(self):
        result=self.compare(self.measured(thread_major_diameter=4),.1)
        self.assertEqual(result['compatible_class_ids'],[1,3])
        self.assertEqual(result['conclusion'],'INSUFFICIENT_EVIDENCE')

    def test_complete_measurement_can_match_other_class_without_relabelling(self):
        measurement=self.measured(thread_major_diameter=4,underhead_length=10)
        result=self.compare(measurement,.1)
        self.assertEqual(result['compatible_class_ids'],[3])
        self.assertEqual(result['conclusion'],'SPEC_COMPATIBLE_CANDIDATE')
        self.assertEqual(measurement['class_id'],2)

    def test_boundary_and_uncertainty_overlap(self):
        result=self.compare(self.measured(underhead_length=11),0)
        row=next(r for r in result['comparisons'] if r['class_id']==2)
        self.assertEqual(row['dimensions']['underhead_length']['state'],'WITHIN')
        result=self.compare(self.measured(underhead_length=11),.2)
        row=next(r for r in result['comparisons'] if r['class_id']==2)
        self.assertEqual(row['dimensions']['underhead_length']['state'],'BORDERLINE')
        self.assertFalse(row['excluded'])

    def test_geometric_candidate_alone_does_not_authorize_exclusion(self):
        result=compare_dimensions(self.measured(thread_major_diameter=9),self.specs)
        self.assertEqual(result['compatible_class_ids'],[1,2,3])
        self.assertEqual(result['conclusion'],'INSUFFICIENT_EVIDENCE')
        result=self.compare(self.measured(thread_major_diameter=9),family_confirmed=False)
        self.assertEqual(result['conclusion'],'INSUFFICIENT_EVIDENCE')

    def test_invalid_specs_measurements_and_quality(self):
        bad=copy.deepcopy(self.specs);bad['items'][0]['class_id']=2
        with self.assertRaises(ValueError):validate_specs(bad)
        with self.assertRaises(ValueError):self.compare(self.measured(thread_major_diameter=float('nan')))
        with self.assertRaises(ValueError):
            compare_dimensions(self.measured(thread_major_diameter=4),self.specs,
                               {'thread_major_diameter':dict(verified=True,max_error_mm=-1,evidence_id='x')})
        self.assertEqual(compare_dimensions(dict(class_id=0,dimensions={}),self.specs)['status'],'REVIEW')

    def calibration(self):
        return dict(calibration_id='synthetic',capture_geometry_id='bench-a',frame_width=100,frame_height=80,
                    px_per_mm=[20,20],validated=False)

    def test_calibration_resolution_and_geometry_guard(self):
        frame=np.zeros((80,100,3),np.uint8)
        observations=_observations([[20.,20.,70.,60.,.8,2]],100,80)
        configs=[None,self.calibration(),dict(self.calibration(),frame_width=200)]
        for cfg,geometry,reason in [(configs[0],'bench-a','CALIBRATION_MISSING'),
                                   (configs[1],'other','CAPTURE_GEOMETRY_MISMATCH'),
                                   (configs[2],'bench-a','CALIBRATION_RESOLUTION_MISMATCH')]:
            processor=SizeEvidenceProcessor(self.specs,cfg)
            with patch('src.p06_size_evidence.measure_object') as measure:
                result=processor.process(frame,observations,capture_geometry_id=geometry)
                self.assertEqual(result['objects'][0]['size_evidence']['reason'],reason)
                measure.assert_not_called()
        with self.assertRaises(ValueError):validate_calibration(dict(self.calibration(),px_per_mm=[0,1]))

    def test_enrichment_preserves_detections_counts_and_frame(self):
        frame=np.full((80,100,3),240,np.uint8);before=frame.copy()
        observations=_observations([[20.,20.,70.,60.,.8,2]],100,80)
        cached=copy.deepcopy(observations)
        processor=SizeEvidenceProcessor(self.specs,self.calibration())
        with patch('src.p06_size_evidence.measure_object',return_value=self.measured(thread_major_diameter=6)):
            result=processor.process(frame,observations,capture_geometry_id='bench-a')
        evidence=result['objects'][0]['size_evidence']
        self.assertEqual(evidence['conclusion'],'INSUFFICIENT_EVIDENCE')
        self.assertEqual(result['counts'],observations['counts'])
        self.assertEqual(result['objects'][0]['class_id'],2)
        self.assertEqual(observations,cached)
        np.testing.assert_array_equal(frame,before)

    def test_real_geometry_entry_and_empty_or_id0_objects(self):
        frame=np.full((80,100,3),240,np.uint8)
        processor=SizeEvidenceProcessor(self.specs,self.calibration())
        result=processor.process(frame,_observations([[20.,20.,70.,60.,.8,2]],100,80),capture_geometry_id='bench-a')
        self.assertEqual(result['objects'][0]['size_evidence']['measurement']['reason'],'NO_FOREGROUND')
        for rows in ([],[[20.,20.,70.,60.,.8,0]]):
            result=processor.process(frame,_observations(rows,100,80),capture_geometry_id='bench-a')
            self.assertEqual(result['counts']['0'],len(rows))
        bad=_observations([],100,80);bad['counts']['1']=1
        with self.assertRaises(ValueError):processor.process(frame,bad,capture_geometry_id='bench-a')

    def test_washer_checks_its_own_two_dimensions(self):
        measured=dict(class_id=5,status='PARTIAL_CANDIDATE',dimensions={'outer_diameter':{'mm_estimate':20}})
        result=self.compare(measured,.1)
        self.assertEqual(result['conclusion'],'OUTSIDE_FAMILY_CANDIDATE')
        self.assertEqual(result['comparisons'][0]['dimensions']['inner_diameter']['state'],'MISSING')

    def test_synthetic_frame_through_crop_geometry_and_comparison(self):
        frame=np.full((500,700,3),240,np.uint8)
        cv2.rectangle(frame,(100,140),(145,345),(25,)*3,-1)
        cv2.rectangle(frame,(145,200),(545,280),(25,)*3,-1)
        cfg=dict(self.calibration(),frame_width=700,frame_height=500)
        result=SizeEvidenceProcessor(self.specs,cfg).process(
            frame,_observations([[95.,135.,550.,350.,.9,1]],700,500),capture_geometry_id='bench-a')
        evidence=result['objects'][0]['size_evidence']
        self.assertIn('thread_major_diameter',evidence['measurement']['dimensions'])
        self.assertEqual(evidence['comparisons'][0]['dimensions']['thread_major_diameter']['state'],'UNVERIFIED')
        self.assertEqual(result['counts']['1'],1)
