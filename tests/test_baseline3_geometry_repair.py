"""Geometric/decision software fixtures; not physical acceptance evidence."""
import unittest
import numpy as np
import cv2
from src.models.baseline3.geometry_repair import measure_object
from src.models.baseline3.size_repaired import assess,foreground_conservation
from src.models.baseline3.size_repaired import RepairedSizeBackend
from types import SimpleNamespace
from src.models.baseline3.size_candidate import DevelopmentForeground
from src.models.baseline3.inference import _observations
from src.p06_foreground import DEFAULTS


class RepairTests(unittest.TestCase):
    def test_bright_internal_stripe_is_not_the_screw_diameter(self):
        image=np.full((400,700,3),180,np.uint8)
        cv2.rectangle(image,(100,160),(480,270),(40,40,40),-1)
        cv2.rectangle(image,(480,105),(520,320),(40,40,40),-1)
        cv2.rectangle(image,(100,198),(480,215),(180,180,180),-1)
        result=measure_object(image,2,[20.,20.],target_box=[90,95,530,330])
        self.assertEqual(result['status'],'MEASURED_CANDIDATE',result['reason'])
        diameter=result['dimensions']['thread_major_diameter']['mm_estimate']
        self.assertGreater(diameter,5.2);self.assertLess(diameter,5.8)
        self.assertTrue(result['diagnostics']['quality_valid_for_rejection'])
        ys=[p[1] for p in result['dimensions']['thread_major_diameter']['line_xy']]
        self.assertGreater(max(ys)-min(ys),104.)

    def test_untrusted_and_borderline_values_do_not_reject_as_id0(self):
        band={'2':{'thread_major_diameter':dict(min_mm=4.6,max_mm=6.4)}}
        result=dict(status='MEASURED_CANDIDATE',dimensions={'thread_major_diameter':dict(mm_estimate=2.63)},
            diagnostics=dict(quality_valid_for_rejection=False,measurement_uncertainty_mm=.1))
        self.assertEqual(assess(result,2,band)[0],'UNMEASURABLE')
        result['diagnostics']['quality_valid_for_rejection']=True
        self.assertEqual(assess(result,2,band)[0],'OUTSIDE_DEVELOPMENT_BAND')
        result['dimensions']['thread_major_diameter']['mm_estimate']=4.55
        self.assertEqual(assess(result,2,band)[0],'BORDERLINE_MEASUREMENT')
        result['dimensions']['thread_major_diameter']['mm_estimate']=5.
        self.assertEqual(assess(result,2,band)[0],'WITHIN_DEVELOPMENT_BAND')

    def test_unmatched_foreground_remains_review_after_fragment_grouping(self):
        bg=np.full((1008,756,3),180,np.uint8)
        fg=DevelopmentForeground(bg,dict(reference_id='SYNTHETIC',capture_geometry_id='FIXTURE',
            parameters=dict(DEFAULTS,difference_threshold=40,minimum_area=30)))
        image=bg.copy();image[400:420,300:380]=40;image[440:460,300:380]=40
        obs=_observations([[295.,395.,385.,465.,.9,1]],756,1008)
        supports={0:[[[295,395],[385,395],[385,465],[295,465]]]}
        result=foreground_conservation(fg,image,obs,supports)
        self.assertTrue(result['counts_complete'],result['reason'])
        image[650:690,100:140]=30
        result=foreground_conservation(fg,image,obs,supports)
        self.assertFalse(result['counts_complete']);self.assertTrue(result['residual_regions'])
        self.assertFalse(result['physical_unknown_validation'])

    def test_overlapping_detection_ownership_cannot_silently_erase_target(self):
        image=np.full((300,500,3),180,np.uint8)
        cv2.rectangle(image,(80,110),(400,190),(40,40,40),-1)
        result=measure_object(image,1,[20.,20.],target_box=[70,100,410,200],excluded_boxes=[[200,100,410,200]])
        self.assertEqual(result['status'],'UNMEASURABLE')
        self.assertFalse(result['diagnostics']['quality_valid_for_rejection'])

    def test_projected_flat_disagreement_is_not_confirmed_wrong_size(self):
        bands={'4':{'across_flats':dict(min_mm=6.3,max_mm=7.7)}}
        m=dict(status='MEASURED_CANDIDATE',dimensions={'across_flats':dict(mm_estimate=6.2)},
            diagnostics=dict(quality_valid_for_rejection=True,measurement_uncertainty_mm=.06,
                             opposite_flats_mm=[6.15,6.2,6.9]))
        self.assertEqual(assess(m,4,bands)[0],'BORDERLINE_MEASUREMENT')
        m['dimensions']['across_flats']['mm_estimate']=10.
        m['diagnostics']['opposite_flats_mm']=[9.8,10.,10.2]
        self.assertEqual(assess(m,4,bands)[0],'OUTSIDE_DEVELOPMENT_BAND')

    def test_conflicting_existing_nominal_does_not_widen_pass_range(self):
        band={'2':{'underhead_length':dict(min_mm=7.3,max_mm=10.08)}}
        nominal=dict(dimensions={'underhead_length':dict(reference_mm=10.,provisional_allowance_mm=1.)})
        m=dict(status='MEASURED_CANDIDATE',dimensions={'underhead_length':dict(mm_estimate=10.4)},
            diagnostics=dict(quality_valid_for_rejection=True,measurement_uncertainty_mm=.06))
        self.assertEqual(assess(m,2,band,nominal)[0],'SPEC_REFERENCE_CONFLICT')
        self.assertEqual(band['2']['underhead_length']['max_mm'],10.08)
        m['dimensions']['underhead_length']['mm_estimate']=40.
        self.assertEqual(assess(m,2,band,nominal)[0],'OUTSIDE_DEVELOPMENT_BAND')

    def test_ambiguous_instance_identity_prevents_size_relabelling(self):
        bg=np.full((1008,756,3),180,np.uint8);frame=bg.copy();frame[400:480,280:400]=40
        native=RepairedSizeBackend.__new__(RepairedSizeBackend)
        native.calibration=dict(frame_size=[756,1008],px_per_mm=[20.,20.],relative_conservative_scale_envelope=.07)
        native.specifications=[];native.detector=SimpleNamespace(_params=dict(max_det=300))
        bands={'underhead_length':dict(min_mm=20.,max_mm=27.),'thread_major_diameter':dict(min_mm=3.3,max_mm=4.3)}
        native.selection=dict(bands={'1':bands},selected={'1':{'parameters':dict(background_offset=25,close_kernel=5,targeted=True)}})
        native.foreground=DevelopmentForeground(bg,dict(reference_id='FIXTURE',capture_geometry_id='FIXTURE',
            parameters=dict(DEFAULTS,difference_threshold=40,minimum_area=30)))
        obs=_observations([[280.,400.,400.,480.,.9,1],[285.,400.,405.,480.,.8,1]],756,1008)
        cache={}
        for o in obs['objects']:
            pad=42;l=int(o['xmin'])-pad;t=int(o['ymin'])-pad
            contour=(np.array([[280,400],[400,400],[400,480],[280,480]])-[l,t]).tolist()
            cache[o['detection_id']]=dict(status='MEASURED_CANDIDATE',reason='',dimensions={
                'underhead_length':dict(mm_estimate=40. if o['detection_id']==0 else 25.,pixels=800.,line_xy=[[50,50],[100,50]]),
                'thread_major_diameter':dict(mm_estimate=4.,pixels=80.,line_xy=[[50,50],[50,100]])},
                contours_xy=[contour],diagnostics=dict(quality_valid_for_rejection=True,measurement_uncertainty_mm=.1))
        result=native.process_observations(frame,obs,{'1':2,'2':0,'3':0,'4':0,'5':0},measurement_cache=cache)
        self.assertEqual(result['status'],'REVIEW');self.assertEqual(result['non_catalogue_count'],0)
        self.assertEqual(result['objects'][0]['class_id'],1)
        self.assertEqual(result['objects'][0]['size_evidence']['conclusion'],'AMBIGUOUS_INSTANCE_SUPPORT')


if __name__=='__main__':unittest.main()
