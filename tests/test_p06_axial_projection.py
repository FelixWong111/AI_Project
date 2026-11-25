import unittest
import cv2
import numpy as np
from tools.p06_axial_projection import project


class ProjectionTests(unittest.TestCase):
    def image(self):
        im=np.full((480,720,3),240,np.uint8)
        cv2.rectangle(im,(100,140),(145,340),(25,)*3,-1)
        cv2.rectangle(im,(145,200),(545,280),(25,)*3,-1)
        return im

    def test_whole_span_includes_head_and_does_not_subtract_head_thickness(self):
        im=self.image();before=im.copy()
        r=project(im,(20,20),(95,135,550,345))
        self.assertAlmostEqual(r['dimension']['mm_estimate'],445/20,delta=.3)
        np.testing.assert_array_equal(im,before)

    def test_anisotropic_scale_and_manual_axis_are_explicit(self):
        im=self.image()
        r=project(im,(20,16),(95,135,550,345),reference_axis=(1,0))
        self.assertAlmostEqual(r['dimension']['mm_estimate'],445/20,delta=.3)
        with self.assertRaises(ValueError):
            project(im,(20,16),(95,135,550,345),reference_axis=(0,0))

    def test_unknown_scale_is_rejected(self):
        with self.assertRaises(ValueError):
            project(self.image(),(0,20),(95,135,550,345))
