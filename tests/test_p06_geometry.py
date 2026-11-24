"""Known synthetic geometry; no real data or product tolerances."""
import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch
import cv2
import numpy as np
from src.p06_geometry import DEFAULT_PARAMETERS, measure_object
from tools.p06_measure_geometry import crop_for_object, result_global, new_output, evaluate, catalogue_records,ruler_crop
from tools.p06_geometry_stage1 import compare
from PIL import Image,ImageDraw


class GeometryTests(unittest.TestCase):
    def image(self):
        return np.full((500, 700, 3), 240, np.uint8)

    def test_screw_underhead_length_excludes_head(self):
        im=self.image()
        cv2.rectangle(im,(100,140),(140,340),(30,30,30),-1)
        cv2.rectangle(im,(140,200),(540,280),(30,30,30),-1)
        r=measure_object(im,1,(20,20))
        self.assertEqual(r['status'],'MEASURED_CANDIDATE',r)
        self.assertAlmostEqual(r['dimensions']['underhead_length']['mm_estimate'],20,delta=.3)
        self.assertAlmostEqual(r['dimensions']['thread_major_diameter']['mm_estimate'],4,delta=.2)
        self.assertEqual(r['decision'],None)
        self.assertAlmostEqual(r['dimensions']['underhead_length']['line_xy'][0][0],140,delta=4)
        self.assertGreaterEqual(r['diagnostics']['paired_edge_sections'],3)
        self.assertTrue(.22 <= r['diagnostics']['shaft_line_fraction'] <= .8)

    def test_hidden_underhead_edge_keeps_only_reviewable_diameter_candidate(self):
        im=self.image()
        cv2.rectangle(im,(100,200),(140,350),(30,30,30),-1)
        cv2.rectangle(im,(140,200),(540,280),(30,30,30),-1)
        r=measure_object(im,1,(20,20),screw_method='stage1')
        self.assertEqual(r['status'],'UNMEASURABLE')
        self.assertTrue(r['reason'].startswith('UNDERHEAD_'),r['reason'])
        self.assertEqual(r['dimensions'],{})
        self.assertTrue(r['diagnostics']['partial_requires_review'])
        self.assertAlmostEqual(r['diagnostics']['partial_dimensions']['thread_major_diameter']['mm_estimate'],4,delta=.25)

    def test_large_head_short_body_does_not_become_shaft_diameter(self):
        im=self.image()
        cv2.rectangle(im,(100,140),(220,340),(30,30,30),-1)
        cv2.rectangle(im,(220,200),(380,280),(30,30,30),-1)
        r=measure_object(im,2,(20,20))
        if r['status']=='MEASURED_CANDIDATE':
            self.assertAlmostEqual(r['dimensions']['thread_major_diameter']['mm_estimate'],4,delta=.25)
            self.assertAlmostEqual(r['dimensions']['underhead_length']['mm_estimate'],8,delta=.3)
        else:
            self.assertEqual(r['dimensions'],{})
            if r['reason']=='SHAFT_SECTION_NOT_CLEAR_OF_HEAD_OR_TIP':
                self.assertNotIn('partial_dimensions',r['diagnostics'])

    def test_audited_endpoints_and_non_nominal_shaft_with_highlight(self):
        im=self.image()
        cv2.rectangle(im,(100,120),(145,345),(22,22,22),-1)
        cv2.rectangle(im,(145,190),(545,295),(30,30,30),-1)
        cv2.rectangle(im,(152,227),(536,250),(225,225,225),-1)
        for cid in (1,2,3):
            r=measure_object(im,cid,(20,20),screw_method='edge_audit')
            self.assertEqual(r['status'],'MEASURED_CANDIDATE',r['reason'])
            a,b=r['dimensions']['underhead_length']['line_xy']
            self.assertAlmostEqual(a[0],145,delta=3)
            self.assertAlmostEqual(b[0],545,delta=3)
            self.assertAlmostEqual(r['dimensions']['thread_major_diameter']['pixels'],105,delta=3)

    def test_audited_shoulder_fails_when_only_one_side_is_visible(self):
        im=self.image()
        cv2.rectangle(im,(100,200),(140,350),(30,30,30),-1)
        cv2.rectangle(im,(140,200),(540,280),(30,30,30),-1)
        r=measure_object(im,1,(20,20),screw_method='edge_audit')
        self.assertEqual(r['reason'],'UNDERHEAD_BASE_NOT_CONFIRMED')
        self.assertFalse(r['dimensions'])

    def test_internal_bright_band_near_tip_does_not_shorten_shaft(self):
        im=self.image()
        cv2.rectangle(im,(100,120),(145,345),(22,)*3,-1)
        cv2.rectangle(im,(145,190),(545,295),(65,)*3,-1)
        cv2.rectangle(im,(528,202),(532,283),(255,)*3,-1)
        r=measure_object(im,2,(20,20),screw_method='edge_audit')
        self.assertEqual(r['status'],'MEASURED_CANDIDATE',r['reason'])
        self.assertAlmostEqual(r['dimensions']['underhead_length']['line_xy'][1][0],545,delta=3)

    def test_radial_aperture_separates_bright_face_from_background(self):
        im=self.image()
        for rad,value in [(150,25),(144,230),(61,25),(55,240)]:
            cv2.circle(im,(350,250),rad,(value,)*3,-1)
        cv2.circle(im,(440,270),20,(245,)*3,-1)
        r=measure_object(im,5,(20,20),radial_washer=True)
        self.assertEqual(r['status'],'MEASURED_CANDIDATE',r['reason'])
        self.assertAlmostEqual(r['dimensions']['inner_diameter']['pixels'],110,delta=3)
        self.assertGreaterEqual(r['diagnostics']['radial_aperture_supported_rays'],54)

    def test_radial_ambiguous_concentric_rims_and_dark_core_refused(self):
        im=self.image()
        for rad,value in [(150,25),(144,232),(70,25),(64,240),(55,25),(49,240)]:
            cv2.circle(im,(350,250),rad,(value,)*3,-1)
        r=measure_object(im,5,(20,20),radial_washer=True)
        self.assertEqual(r['status'],'UNMEASURABLE',r)
        cv2.circle(im,(350,250),45,(120,)*3,-1)
        r=measure_object(im,5,(20,20),radial_washer=True)
        self.assertEqual(r['reason'],'APERTURE_CENTER_BACKGROUND_UNCONFIRMED')

    def test_degenerate_washer_hull_is_reported_without_opencv_exception(self):
        im=self.image();cv2.rectangle(im,(200,100),(500,400),(20,20,20),-1)
        cv2.circle(im,(350,250),50,(240,240,240),-1)
        r=measure_object(im,5,(20,20))
        self.assertEqual(r['status'],'UNMEASURABLE')
        self.assertIn(r['reason'],('ELLIPSE_SUPPORT_DEGENERATE','CIRCULAR_EDGE_ANGULAR_COVERAGE_LOW'))

    def test_rotated_screw_and_original_pixel_lines(self):
        im=self.image()
        cv2.rectangle(im,(180,150),(210,330),(20,20,20),-1)
        cv2.rectangle(im,(210,200),(510,280),(20,20,20),-1)
        matrix=cv2.getRotationMatrix2D((350,250),37,1)
        im=cv2.warpAffine(im,matrix,(700,500),borderValue=(240,240,240))
        r=measure_object(im,3,(20,20))
        self.assertEqual(r['status'],'MEASURED_CANDIDATE',r)
        self.assertAlmostEqual(r['dimensions']['underhead_length']['mm_estimate'],15,delta=.5)
        for dim in r['dimensions'].values():
            a,b=np.array(dim['line_xy'])
            self.assertAlmostEqual(np.linalg.norm(a-b),dim['pixels'],places=5)
            self.assertAlmostEqual(dim['pixels']/20,dim['mm_estimate'],places=5)

    def test_head_at_opposite_end(self):
        im=self.image();cv2.rectangle(im,(100,140),(140,340),(30,30,30),-1)
        cv2.rectangle(im,(140,200),(540,280),(30,30,30),-1)
        r=measure_object(cv2.flip(im,1),2,(20,20))
        self.assertEqual(r['status'],'MEASURED_CANDIDATE',r)
        self.assertAlmostEqual(r['dimensions']['underhead_length']['mm_estimate'],20,delta=.3)

    def test_soft_shadow_does_not_define_thread_diameter(self):
        im=self.image()
        shadow=np.zeros(im.shape[:2],np.uint8)
        cv2.rectangle(shadow,(90,160),(550,315),160,-1)
        shadow=cv2.GaussianBlur(shadow,(41,41),10)
        im[:]=np.maximum(im.astype(int)-shadow[:,:,None],0).astype(np.uint8)
        cv2.rectangle(im,(100,120),(140,350),(20,20,20),-1)
        cv2.rectangle(im,(140,200),(540,280),(20,20,20),-1)
        r=measure_object(im,1,(20,20))
        self.assertEqual(r['status'],'MEASURED_CANDIDATE',r)
        self.assertAlmostEqual(r['dimensions']['thread_major_diameter']['mm_estimate'],4,delta=.25)
        audited=measure_object(im,1,(20,20),screw_method='edge_audit')
        if audited['status']=='MEASURED_CANDIDATE':
            self.assertAlmostEqual(audited['dimensions']['thread_major_diameter']['pixels'],80,delta=5)
        else:
            self.assertFalse(audited['dimensions'])

    def test_washer_inner_outer_and_bright_material(self):
        for metal in (30,205):
            im=self.image()
            cv2.circle(im,(350,250),150,(metal,metal,metal),-1)
            cv2.circle(im,(350,250),55,(240,240,240),-1)
            r=measure_object(im,5,(20,20))
            self.assertEqual(r['status'],'MEASURED_CANDIDATE',r)
            self.assertAlmostEqual(r['dimensions']['outer_diameter']['mm_estimate'],15,delta=.2)
            self.assertAlmostEqual(r['dimensions']['inner_diameter']['mm_estimate'],5.5,delta=.2)

    def test_bright_washer_with_nested_dark_rims(self):
        im=self.image()
        cv2.circle(im,(350,250),150,(25,25,25),-1)
        cv2.circle(im,(350,250),144,(235,235,235),-1)
        cv2.circle(im,(350,250),61,(25,25,25),-1)
        cv2.circle(im,(350,250),55,(240,240,240),-1)
        r=measure_object(im,5,(20,20))
        self.assertEqual(r['status'],'MEASURED_CANDIDATE',r)
        self.assertAlmostEqual(r['dimensions']['outer_diameter']['mm_estimate'],15,delta=.2)
        self.assertAlmostEqual(r['dimensions']['inner_diameter']['mm_estimate'],5.5,delta=.2)
        self.assertGreater(r['diagnostics']['aperture_hierarchy_depth'],1)

    def test_bright_offcentre_highlight_is_not_selected_as_hole(self):
        im=self.image();cv2.circle(im,(350,250),150,(25,25,25),-1)
        cv2.circle(im,(350,250),55,(240,240,240),-1)
        cv2.circle(im,(440,270),35,(240,240,240),-1)
        r=measure_object(im,5,(20,20))
        self.assertEqual(r['status'],'MEASURED_CANDIDATE',r)
        self.assertAlmostEqual(r['dimensions']['inner_diameter']['mm_estimate'],5.5,delta=.2)

    def test_serrated_thread_crest_with_internal_bright_stripe_and_shadow(self):
        im=self.image();shadow=np.zeros(im.shape[:2],np.uint8)
        cv2.rectangle(shadow,(150,150),(560,310),120,-1)
        im-=cv2.GaussianBlur(shadow,(41,41),10)[:,:,None]
        cv2.rectangle(im,(100,130),(145,350),(20,20,20),-1)
        cv2.rectangle(im,(140,208),(540,272),(20,20,20),-1)
        for x in range(145,530,16):
            cv2.rectangle(im,(x,200),(x+7,280),(20,20,20),-1)
        cv2.rectangle(im,(150,230),(535,245),(235,235,235),-1)
        r=measure_object(im,1,(20,20))
        self.assertEqual(r['status'],'MEASURED_CANDIDATE',r)
        self.assertAlmostEqual(r['dimensions']['thread_major_diameter']['mm_estimate'],4,delta=.25)
        strict=measure_object(im,1,(20,20),screw_method='stage1')
        self.assertIn('section_widths_mm',strict['diagnostics'])

    def test_nut_opposite_flats_not_corner_diameter(self):
        im=self.image();radius=140/np.sqrt(3)
        polygon=np.array([(350+radius*np.cos(a),250+radius*np.sin(a))
                          for a in np.arange(6)*np.pi/3+.2],np.int32)
        cv2.fillPoly(im,[polygon],(25,25,25));cv2.circle(im,(350,250),30,(240,240,240),-1)
        r=measure_object(im,4,(20,20))
        self.assertEqual(r['status'],'MEASURED_CANDIDATE',r)
        self.assertAlmostEqual(r['dimensions']['across_flats']['mm_estimate'],7,delta=.2)
        self.assertEqual(len(r['diagnostics']['opposite_flats_mm']),3)

    def test_no_head_no_hole_and_not_hexagon_rejected(self):
        im=self.image();cv2.rectangle(im,(160,210),(540,290),(30,30,30),-1)
        self.assertEqual(measure_object(im,1,(20,20))['status'],'UNMEASURABLE')
        im=self.image();cv2.circle(im,(350,250),130,(30,30,30),-1)
        self.assertEqual(measure_object(im,5,(20,20))['status'],'UNMEASURABLE')
        self.assertEqual(measure_object(im,4,(20,20))['status'],'UNMEASURABLE')

    def test_empty_touching_and_unknown_class_fail_explicitly(self):
        im=self.image()
        self.assertEqual(measure_object(im,2,(20,20))['reason'],'NO_FOREGROUND')
        cv2.rectangle(im,(0,100),(300,300),(30,30,30),-1)
        self.assertEqual(measure_object(im,2,(20,20))['reason'],'FOREGROUND_TOUCHES_CROP')
        self.assertEqual(measure_object(im,0,(20,20))['reason'],'NON_CATALOGUE_NOT_MEASURED')
        with self.assertRaises(ValueError):measure_object(im,6,(20,20))

    def test_anisotropic_scale_and_input_unchanged(self):
        im=self.image()
        cv2.ellipse(im,(350,250),(150,120),0,0,360,(20,20,20),-1)
        cv2.ellipse(im,(350,250),(50,40),0,0,360,(240,240,240),-1)
        before=im.copy();params=dict(DEFAULT_PARAMETERS)
        r=measure_object(im,5,(20,16),params)
        self.assertEqual(r['status'],'MEASURED_CANDIDATE',r)
        self.assertAlmostEqual(r['dimensions']['outer_diameter']['mm_estimate'],15,delta=.25)
        np.testing.assert_array_equal(im,before)
        self.assertEqual(params,DEFAULT_PARAMETERS)
        for scale in ((0,1),(float('nan'),1)):
            with self.assertRaises(ValueError):measure_object(im,5,scale)

    def test_multiple_objects_and_excessive_tilt_rejected(self):
        im=self.image()
        cv2.circle(im,(180,250),80,(20,20,20),-1)
        cv2.circle(im,(500,250),80,(20,20,20),-1)
        self.assertEqual(measure_object(im,5,(20,20))['status'],'UNMEASURABLE')
        im=self.image()
        cv2.ellipse(im,(350,250),(160,70),0,0,360,(20,20,20),-1)
        cv2.ellipse(im,(350,250),(55,24),0,0,360,(240,240,240),-1)
        self.assertEqual(measure_object(im,5,(20,20))['status'],'UNMEASURABLE')

    def test_crop_origin_translation_preserves_measurement(self):
        image=Image.new('RGB',(600,400),'white')
        row=dict(xmin='100',ymin='80',xmax='300',ymax='250')
        crop,box=crop_for_object(image,row)
        self.assertEqual(crop.size,(box[2]-box[0],box[3]-box[1]))
        r=dict(dimensions={'length':dict(line_xy=[[5,10],[55,10]],pixels=50,mm_estimate=2.5)},
               diagnostics={'partial_dimensions':{'thread_major_diameter':dict(line_xy=[[10,5],[10,25]],pixels=20,mm_estimate=1)}},
               contours_xy=[[[5,5],[55,5],[55,20],[5,20]]])
        translated=result_global(r,(100,200))
        self.assertEqual(translated['dimensions']['length']['line_xy'],[[105,210],[155,210]])
        self.assertEqual(translated['dimensions']['length']['pixels'],50)
        self.assertEqual(r['dimensions']['length']['line_xy'][0],[5,10])
        self.assertEqual(translated['diagnostics']['partial_dimensions']['thread_major_diameter']['line_xy'],[[110,205],[110,225]])

    def test_ruler_example_is_cropped_around_object_not_background_center(self):
        image=Image.new('RGB',(3024,4032),'white')
        ImageDraw.Draw(image).ellipse((900,2500,1100,2700),fill='black')
        crop,box=ruler_crop(image)
        self.assertLess(abs((box[0]+box[2])/2-1000),2)
        self.assertLess(abs((box[1]+box[3])/2-2600),2)
        self.assertLess(crop.width,300)
        self.assertEqual(image.size,(3024,4032))

    def test_no_test_argument_output_guard_and_changed_freeze_refusal(self):
        with self.assertRaises(ValueError):catalogue_records('test')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            with patch('tools.p06_measure_geometry.ROOT',root):
                with self.assertRaises(ValueError):new_output(root/'outside')
                (root/'output').mkdir();(root/'output/existing').mkdir()
                with self.assertRaises(ValueError):new_output(root/'output/existing')
            frozen=root/'freeze.json';frozen.write_text('{"module_sha256":"changed"}')
            with patch('tools.p06_measure_geometry.catalogue_records') as read:
                with self.assertRaisesRegex(ValueError,'Code changed'):evaluate(frozen,root/'derived')
                read.assert_not_called()

    def test_stage1_selection_cannot_access_a_non_train_image(self):
        with patch('tools.p06_geometry_stage1.before_module'), \
             patch('tools.p06_geometry_stage1.catalogue_records',return_value=({},{})) as inventory:
            with self.assertRaisesRegex(ValueError,'non-Train'):
                compare({'train_image_selection':['outside.jpg']})
            inventory.assert_called_once_with('train')

    def test_invalid_parameters_rejected_without_mutating_defaults(self):
        for params in ({}, {**DEFAULT_PARAMETERS,'close_kernel':4},
                       {**DEFAULT_PARAMETERS,'background_offset':float('nan')}):
            with self.assertRaises(ValueError):measure_object(self.image(),1,(20,20),params)


if __name__=='__main__':unittest.main()
