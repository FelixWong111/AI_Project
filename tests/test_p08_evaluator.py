import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.evaluator import evaluate


def csvfile(path, columns, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


class EvaluatorTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.pred = self.root / 'pred'
        self.pred.mkdir()
        self.images = self.root / 'gt_images.csv'
        self.objects = self.root / 'gt_objects.csv'
        self.expected = self.root / 'expected.json'
        self.ids = ['empty.jpg', 'miss.jpg', 'foreign.jpg', 'ignore.jpg', 'review.jpg']
        csvfile(self.images, ['image_id','scene_id','split','width','height','annotation_status'], [
            dict(image_id=i, scene_id=i, split='test', width=100, height=100,
                 annotation_status='REVIEW' if i == 'review.jpg' else 'OK') for i in self.ids])
        objcols = ['image_id','object_id','class_id','xmin','ymin','xmax','ymax','ignore']
        csvfile(self.objects, objcols, [
            dict(image_id='miss.jpg', object_id='a', class_id=1, xmin=0, ymin=0, xmax=10, ymax=10, ignore=0),
            dict(image_id='foreign.jpg', object_id='a', class_id=0, xmin=20, ymin=20, xmax=30, ymax=30, ignore=0),
            dict(image_id='ignore.jpg', object_id='a', class_id=1, xmin=10, ymin=10, xmax=20, ymax=20, ignore=1)])
        self.expected.write_text(json.dumps([{'image_id': i, 'expected_counts': {str(c): 0 for c in range(1,6)}} for i in self.ids]))

    def test_image_level_confusion_and_exclusions(self):
        labels = ['PASS','PASS','FAIL','PASS','PASS']
        csvfile(self.pred/'images.csv', ['image_id','model_id','status','predicted_label','prediction_score','score_type','non_catalogue_count'], [
            dict(image_id=i, model_id='baseline1', status=l, predicted_label=l, prediction_score=.8, score_type='prob', non_catalogue_count='')
            for i, l in zip(self.ids, labels)])
        result = evaluate(self.images, self.objects, self.expected, self.pred, self.root/'out')
        self.assertEqual((result['image_level']['tp'], result['image_level']['fp'], result['image_level']['fn'], result['image_level']['tn']), (1, 0, 1, 1))
        self.assertEqual(result['scored_image_cases'], 3)
        self.assertEqual(result['object_classes'], {})

    def test_detection_matching_and_ignore(self):
        # Correct foreign object, duplicate detection, and a prediction overlapping ignored GT.
        rows = [dict(image_id='foreign.jpg', model_id='baseline3', detection_id=i, class_id=0,
                     xmin=20, ymin=20, xmax=30, ymax=30, score=score) for i, score in [(0,.9),(1,.8)]]
        rows += [dict(image_id='ignore.jpg', model_id='baseline3', detection_id=0, class_id=1,
                      xmin=10, ymin=10, xmax=20, ymax=20, score=.9)]
        csvfile(self.pred/'objects.csv', ['image_id','model_id','detection_id','class_id','xmin','ymin','xmax','ymax','score'], rows)
        actual = {'foreign.jpg': {0: 2}, 'ignore.jpg': {1: 1}}
        csvfile(self.pred/'counts.csv', ['image_id','class_id','observed_count','expected_count','difference'], [
            dict(image_id=i, class_id=c, observed_count=actual.get(i,{}).get(c,0), expected_count=0,
                 difference=actual.get(i,{}).get(c,0)) for i in self.ids for c in range(6)])
        csvfile(self.pred/'images.csv', ['image_id','model_id','status','predicted_label','prediction_score','score_type','non_catalogue_count'], [
            dict(image_id=i, model_id='baseline3', status='FAIL' if actual.get(i) else 'PASS',
                 predicted_label='FAIL' if actual.get(i) else 'PASS', prediction_score='', score_type='',
                 non_catalogue_count=actual.get(i,{}).get(0,0)) for i in self.ids])
        result = evaluate(self.images, self.objects, self.expected, self.pred, self.root/'out')
        self.assertEqual((result['object_classes']['0']['tp'],result['object_classes']['0']['fp']), (1,1))
        self.assertEqual(result['object_classes']['1']['ignored_predictions'], 1)
        self.assertEqual(result['object_classes']['1']['fn'], 1)
        self.assertAlmostEqual(result['object_classes']['0']['count_mae'], 1/3)
        self.assertAlmostEqual(result['count_vector_exact_rate'], 2/3)

    def test_rejects_missing_prediction(self):
        csvfile(self.pred/'images.csv', ['image_id','model_id','status','predicted_label','prediction_score','score_type','non_catalogue_count'], [])
        with self.assertRaisesRegex(ValueError, 'do not match'):
            evaluate(self.images, self.objects, self.expected, self.pred, self.root/'out')


if __name__ == '__main__':
    unittest.main()
