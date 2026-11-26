"""Comparison infrastructure checks on synthetic fixtures, not held-out photographs."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from src.evaluator import image_level_details


class ComparisonPreparationTests(unittest.TestCase):
    def test_false_pass_denominator_and_unresolved_coverage(self):
        cases=[('FAIL','FAIL'),('FAIL','PASS'),('FAIL','REVIEW'),('PASS','FAIL'),('PASS','PASS'),('PASS','ERROR')]
        rows=[dict(ground_truth_label=gt,prediction_status=status,predicted_label=status if status in ('PASS','FAIL') else '') for gt,status in cases]
        result=image_level_details(rows)
        self.assertEqual(result['confusion_matrix']['rows_truth_PASS_FAIL'],[[1,1],[1,1]])
        self.assertEqual(result['false_pass']['denominator'],3)
        self.assertAlmostEqual(result['false_pass']['rate'],1/3)
        self.assertAlmostEqual(result['prediction_coverage'],4/6)
        self.assertAlmostEqual(result['all_input_correct_fraction'],2/6)
        self.assertEqual(result['completed_accuracy'],.5)
        self.assertEqual(result['unresolved_eligible'],2)

    def test_unknown_ground_truth_does_not_become_a_class_label(self):
        rows=[dict(ground_truth_label='',prediction_status='PASS',predicted_label='PASS')]
        result=image_level_details(rows)
        self.assertEqual(result['ground_truth_eligible'],0)
        self.assertIsNone(result['completed_accuracy'])
        self.assertEqual(result['prediction_coverage'],1)
        self.assertIsNone(result['eligible_coverage'])

    def test_all_fail_predictor_retains_zero_pass_recall(self):
        rows=[dict(ground_truth_label=label,prediction_status='FAIL',predicted_label='FAIL') for label in ('PASS','FAIL')]
        result=image_level_details(rows)
        self.assertEqual(result['passed']['recall'],0)
        self.assertEqual(result['false_fail']['rate'],1)
        self.assertEqual(result['completed_balanced_accuracy'],.5)

    def test_p08_explicit_config_reaches_batch_command(self):
        from p08_run import run
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder);data=base/'fixture';data.mkdir();(data/'images').mkdir()
            for name in ('ground_truth_images.csv','ground_truth_objects.csv','expected_counts.json'):(data/name).write_text('fixture')
            profile=base/'independent.json';profile.write_text('{}')
            with patch('p08_run.subprocess.run',return_value=SimpleNamespace(returncode=0,stdout='synthetic command test')),patch('p08_run.evaluate',return_value={'scored_image_cases':0}):
                result=run(data,'baseline1',base/'out',model_config=profile)
            command=json.loads((result/'command.json').read_text(encoding='utf-8'))
            index=command['command'].index('--config')
            self.assertEqual(command['command'][index+1],str(profile.resolve()))
            self.assertEqual(command['model_config'],str(profile.resolve()))


if __name__=='__main__':unittest.main()
