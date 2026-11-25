import unittest
import cv2
import numpy as np
from src.p06_projection import axial_projection


class ProjectionTests(unittest.TestCase):
    def screw(self):
        image=np.full((400,650,3),240,np.uint8)
        cv2.rectangle(image,(100,110),(140,290),(30,)*3,-1)
        cv2.rectangle(image,(140,165),(540,235),(30,)*3,-1)
        return image

    def test_includes_head_without_subtracting_assumed_thickness(self):
        image=self.screw();before=image.copy()
        r=axial_projection(image,(95,105,545,295),(20,20))
        self.assertEqual(r['status'],'PROJECTED_CANDIDATE',r)
        self.assertAlmostEqual(r['mm_estimate'],22,delta=.3)
        self.assertIsNone(r['decision'])
        np.testing.assert_array_equal(before,image)

    def test_rotation_and_anisotropic_scale(self):
        matrix=cv2.getRotationMatrix2D((325,200),23,1)
        image=cv2.warpAffine(self.screw(),matrix,(650,400),borderValue=(240,)*3)
        r=axial_projection(image,(30,20,610,380),(20,20))
        self.assertEqual(r['status'],'PROJECTED_CANDIDATE',r)
        self.assertAlmostEqual(r['mm_estimate'],22,delta=.5)
        p=np.asarray(r['line_xy'])
        self.assertAlmostEqual(np.linalg.norm(p[1]-p[0])/20,r['mm_estimate'])

    def test_empty_and_invalid_scale(self):
        image=np.full((300,400,3),240,np.uint8)
        self.assertEqual(axial_projection(image,(10,10,390,290),(20,20))['status'],'UNMEASURABLE')
        with self.assertRaises(ValueError):
            axial_projection(image,(10,10,390,290),(0,20))
