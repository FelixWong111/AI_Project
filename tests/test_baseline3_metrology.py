"""Software contract tests only; synthetic fixtures are not physical acceptance."""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

from src.models.baseline3.metrology import fit_metric_calibration,ruler_interval_errors
from src.models.baseline3.size_candidate import size_conclusion,SizeDevelopmentBackend,DevelopmentForeground
from src.models.baseline3.inference import _observations
from src.p06_foreground import DEFAULTS
from src.p07_bridge import validate_result


class MetrologyContracts(unittest.TestCase):
    def test_metric_interval_is_independent_of_arbitrary_ruler_zero(self):
        ruler=dict(image_id='SYNTHETIC',px_per_mm=35.,points=[
            dict(x=100.+i*35,y=400.,relative_mm=i+27,used=True) for i in range(25)])
        fit=fit_metric_calibration([ruler],mode='global')
        errors=ruler_interval_errors(ruler,fit['matrix_pixel_to_mm'])
        self.assertLess(errors['max_absolute_error_mm'],1e-6)
        self.assertEqual(errors['intervals'],15)

    def test_measured_dimensions_are_required_for_size_pass(self):
        bands={'1':{'underhead_length':dict(min_mm=20.,max_mm=27.),
                    'thread_major_diameter':dict(min_mm=3.3,max_mm=4.3)}}
        measured=dict(status='MEASURED_CANDIDATE',dimensions={
            'underhead_length':dict(mm_estimate=25.),'thread_major_diameter':dict(mm_estimate=4.)})
        self.assertEqual(size_conclusion(measured,1,bands)[0],'WITHIN_DEVELOPMENT_BAND')
        measured['dimensions']['underhead_length']['mm_estimate']=40.
        self.assertEqual(size_conclusion(measured,1,bands)[0],'OUTSIDE_DEVELOPMENT_BAND')
        measured['dimensions'].pop('underhead_length');measured['status']='PARTIAL_CANDIDATE'
        self.assertEqual(size_conclusion(measured,1,bands)[0],'UNMEASURABLE')

    def backend(self):
        backend=SizeDevelopmentBackend.__new__(SizeDevelopmentBackend)
        backend.detector=SimpleNamespace(_params=dict(max_det=300))
        return backend

    def observation(self):
        obs=_observations([[i*20+1,1,i*20+15,15,.9,i+1] for i in range(5)],200,200)
        obs['inspection']=dict(failure_reasons=[],review_reasons=[])
        return obs

    def test_expected_changes_only_business_decision_and_differences(self):
        b=self.backend();obs=self.observation();before=copy.deepcopy(obs)
        fixed=dict.fromkeys('12345',1)
        passed=b.rejudge(obs,fixed);failed=b.rejudge(passed,dict(fixed,**{'5':2}));restored=b.rejudge(failed,fixed)
        self.assertEqual([passed['status'],failed['status'],restored['status']],['PASS','FAIL','PASS'])
        for result in (passed,failed,restored):
            self.assertEqual(result['objects'],obs['objects']);self.assertIsNone(result['prediction_score'])
            validate_result(result,'baseline3',(200,200,3))
        self.assertEqual(passed['counts'],failed['counts']);self.assertEqual(obs,before)

    def test_id0_forces_fail_even_when_all_catalogue_counts_match(self):
        obs=self.observation();obs['objects'][0]['class_id']=0
        obs['inspection']['review_reasons']=['measurement incomplete']
        expected=dict.fromkeys('12345',1);expected['1']=0
        result=self.backend().rejudge(obs,expected)
        self.assertEqual(result['status'],'FAIL');self.assertEqual(result['non_catalogue_count'],1)

    def test_genuine_uncertainty_prevents_pass(self):
        obs=self.observation();obs['inspection']['review_reasons']=['CALIBRATION_FRAME_MISMATCH']
        result=self.backend().rejudge(obs,dict.fromkeys('12345',1))
        self.assertEqual(result['status'],'REVIEW');self.assertIsNone(result['predicted_label'])

    def test_foreground_does_not_fabricate_unknown_classification(self):
        background=np.full((1008,756,3),180,np.uint8)
        fg=DevelopmentForeground(background,dict(reference_id='SYNTHETIC_EMPTY',capture_geometry_id='FIXTURE',
            parameters=dict(DEFAULTS,difference_threshold=30,minimum_area=30)))
        image=background.copy();image[400:450,300:350]=30
        result=fg.process(image,_observations([],756,1008))
        self.assertFalse(result['counts_complete'])
        self.assertTrue(any(c['reason']=='UNMATCHED_FOREGROUND' for c in result['candidates']))
        self.assertFalse(any(c['class_id']==0 for c in result['candidates']))
        self.assertFalse(result['physical_unknown_validation'])

    def test_dashboard_shows_actual_development_band_and_unvalidated_error(self):
        from src import p07_dashboard as dashboard
        observation=self.observation();observation['objects'][0]['size_evidence']=dict(
            development_acceptance_bands={'thread_major_diameter':dict(min_mm=3.3,max_mm=4.3)},
            quality=dict(ruler_scale_envelope=.073))
        shown=[]
        with patch.object(dashboard,'text',side_effect=lambda canvas,value,*args,**kwargs:shown.append(str(value))):
            dashboard.draw_diagnostics(np.zeros((1000,1600,3),np.uint8),observation)
        self.assertTrue(any('Development band thread_major_diameter: 3.300..4.300' in s for s in shown))
        self.assertTrue(any('physical error bound NOT VALIDATED' in s for s in shown))


if __name__=='__main__':unittest.main()
