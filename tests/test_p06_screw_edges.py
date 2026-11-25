import unittest
import cv2
import numpy as np
from src.p06_geometry import measure_object


class BoundedRefinementTests(unittest.TestCase):
    def test_non_nominal_dimensions_are_not_snapped_to_class(self):
        im=np.full((500,700,3),240,np.uint8)
        cv2.rectangle(im,(100,120),(145,355),(25,)*3,-1)
        cv2.rectangle(im,(145,195),(545,300),(25,)*3,-1)
        before=im.copy()
        results=[measure_object(im,cid,(20,20),screw_method='bounded_refine') for cid in (1,2,3)]
        for result in results:
            self.assertEqual(result['status'],'MEASURED_CANDIDATE',result)
            self.assertAlmostEqual(result['dimensions']['thread_major_diameter']['pixels'],105,delta=4)
            self.assertAlmostEqual(result['dimensions']['underhead_length']['pixels'],400,delta=5)
        self.assertEqual(results[0]['dimensions'],results[1]['dimensions'])
        self.assertEqual(results[1]['dimensions'],results[2]['dimensions'])
        np.testing.assert_array_equal(im,before)

    def test_empty_image_has_no_fabricated_dimensions(self):
        r=measure_object(np.full((100,100,3),240,np.uint8),1,(20,20),screw_method='bounded_refine')
        self.assertFalse(r['dimensions'])
        self.assertEqual(r['status'],'UNMEASURABLE')
