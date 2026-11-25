"""Synthetic scenes verify bookkeeping and ambiguity, not physical accuracy."""
import copy
import unittest
import cv2
import numpy as np
from src.p06_foreground import ForegroundAssociator,DEFAULTS
from src.models.baseline3.inference import _observations


class ForegroundTests(unittest.TestCase):
    def setUp(self):
        self.bg=np.full((200,300,3),240,np.uint8)
        self.frame=self.bg.copy()
        cv2.rectangle(self.frame,(50,50),(90,100),(30,)*3,-1)
        self.engine=ForegroundAssociator(self.bg,reference_id='synthetic-empty',capture_geometry_id='fixture')

    def run_scene(self,rows,frame=None):
        return self.engine.process(self.frame if frame is None else frame,
                                   _observations(rows,300,200),capture_geometry_id='fixture')

    def test_single_object_and_raw_counts_preserved(self):
        r=self.run_scene([[48.,48.,93.,103.,.9,2]])
        self.assertEqual(r['proposed_counts']['2'],1)
        self.assertTrue(r['counts_complete'])
        self.assertEqual(r['candidates'][0]['reason'],'ASSOCIATED')

    def test_unmatched_foreground_stays_unknown_instead_of_id0(self):
        r=self.run_scene([])
        self.assertEqual(r['candidates'][0]['reason'],'UNMATCHED_FOREGROUND')
        self.assertIsNone(r['candidates'][0]['class_id'])
        self.assertFalse(r['counts_complete'])
        self.assertEqual(r['proposed_counts']['0'],0)

    def test_duplicate_same_class_is_grouped_once(self):
        r=self.run_scene([[48.,48.,93.,103.,.9,2],[49.,49.,92.,102.,.7,2]])
        self.assertEqual(r['raw_counts']['2'],2)
        self.assertEqual(r['proposed_counts']['2'],1)
        self.assertEqual(r['suppressed_detection_ids'],[1])

    def test_cross_class_duplicate_retains_class_ambiguity(self):
        r=self.run_scene([[48.,48.,93.,103.,.9,2],[49.,49.,92.,102.,.7,3]])
        self.assertEqual(r['candidates'][0]['reason'],'CROSS_CLASS_DUPLICATE_AMBIGUOUS')
        self.assertFalse(r['counts_complete'])
        self.assertEqual(r['suppressed_detection_ids'],[])

    def test_two_separate_objects_counted_separately(self):
        cv2.rectangle(self.frame,(180,50),(220,100),(30,)*3,-1)
        r=self.run_scene([[48.,48.,93.,103.,.9,2],[178.,48.,223.,103.,.8,2]])
        self.assertEqual(len(r['candidates']),2)
        self.assertEqual(r['proposed_counts']['2'],2)

    def test_one_box_covering_two_regions_is_ambiguous(self):
        cv2.rectangle(self.frame,(180,50),(220,100),(30,)*3,-1)
        r=self.run_scene([[48.,48.,223.,103.,.9,2]])
        self.assertEqual(r['candidates'][0]['reason'],'ONE_DETECTION_OR_CHAIN_LINKS_MULTIPLE_REGIONS')
        self.assertFalse(r['counts_complete'])

    def test_touching_objects_are_not_automatically_suppressed(self):
        cv2.rectangle(self.frame,(90,50),(130,100),(30,)*3,-1)
        r=self.run_scene([[48.,48.,93.,103.,.9,2],[88.,48.,133.,103.,.8,2]])
        self.assertEqual(r['candidates'][0]['reason'],'MULTIPLE_OBJECTS_OR_MERGED_FOREGROUND')
        self.assertEqual(r['suppressed_detection_ids'],[])

    def test_global_lighting_change_requests_review(self):
        r=self.run_scene([[48.,48.,93.,103.,.9,2]],np.zeros_like(self.frame))
        self.assertFalse(r['background_usable'])
        self.assertFalse(r['counts_complete'])
        self.assertTrue(all(g['reason']=='BACKGROUND_OR_SCENE_CHANGED' for g in r['candidates']))

    def test_roi_border_object_requests_review(self):
        self.engine=ForegroundAssociator(self.bg,reference_id='x',capture_geometry_id='fixture',roi=(50,40,150,130))
        r=self.run_scene([[48.,48.,93.,103.,.9,2]])
        self.assertEqual(r['candidates'][0]['reason'],'FOREGROUND_TOUCHES_ROI_EDGE')

    def test_blank_and_tiny_noise(self):
        im=self.bg.copy();im[40,40]=0
        r=self.run_scene([],im)
        self.assertEqual(r['regions'],[])
        self.assertEqual(r['proposed_counts'],dict.fromkeys('012345',0))

    def test_detection_without_support_is_preserved(self):
        r=self.run_scene([[48.,48.,93.,103.,.9,0]],self.bg)
        self.assertEqual(r['candidates'][0]['reason'],'DETECTION_WITHOUT_FOREGROUND_SUPPORT')
        self.assertEqual(r['raw_counts']['0'],1)
        self.assertFalse(r['counts_complete'])

    def test_inputs_not_modified_and_bad_observations_rejected(self):
        observations=_observations([[48.,48.,93.,103.,.9,2]],300,200)
        old=copy.deepcopy(observations);frame=self.frame.copy()
        r=self.engine.process(self.frame,observations,capture_geometry_id='fixture')
        r['detector_observations']['objects'][0]['score']=0
        self.assertEqual(old,observations)
        np.testing.assert_array_equal(frame,self.frame)
        observations['counts']['2']=True
        with self.assertRaises(ValueError):self.engine.process(self.frame,observations,capture_geometry_id='fixture')

    def test_reference_and_geometry_guards(self):
        with self.assertRaises(ValueError):ForegroundAssociator(self.bg,reference_id='',capture_geometry_id='fixture')
        with self.assertRaises(ValueError):ForegroundAssociator(self.bg,reference_id='x',capture_geometry_id='fixture',config={**DEFAULTS,'kernel_size':2})
        for im,geo in [(self.frame,'different'),(self.frame[:100],'fixture')]:
            with self.assertRaises(ValueError):
                self.engine.process(im,_observations([],300,200),capture_geometry_id=geo)
