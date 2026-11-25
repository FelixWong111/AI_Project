"""Synthetic fixtures only; no project images or held-out data are used."""
import csv
import json
import math
import random
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from src.data.p06_labelme import CLASSES, DataError, inspect_dataset, to_yolo, from_yolo, export_dataset


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.input = self.root / 'input'
        self.input.mkdir()

    def sample(self, name='a', label='long_screw', points=None, **updates):
        Image.new('RGB', (200, 100), (len(name), 10, 20)).save(self.input / (name + '.png'))
        doc = dict(imagePath=name + '.png', imageWidth=200, imageHeight=100,
                   shapes=[dict(label=label, shape_type='rectangle',
                                points=points or [[20, 10], [100, 60]], flags={})], flags={})
        doc.update(updates)
        path = self.input / (name + '.json')
        path.write_text(json.dumps(doc), encoding='utf-8')
        return path

    def manifest(self, rows):
        path = self.root / 'manifest.csv'
        with path.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(['image_id', 'scene_id', 'split'])
            writer.writerows(rows)
        return path

    def report(self, **kwargs):
        return inspect_dataset(self.input, **kwargs)

    def assert_error(self, code, **kwargs):
        report = self.report(**kwargs)
        self.assertFalse(report['ok'], report)
        self.assertIn(code, [e['type'] for e in report['errors']], report)
        for error in report['errors']:
            self.assertTrue(error['file'])
            self.assertTrue(error['suggestion'])

    def test_six_classes(self):
        self.assertEqual(CLASSES, ('non_catalogue', 'long_screw', 'thick_screw',
                                  'thin_screw', 'hex_nut', 'flat_washer'))
        for cid, label in enumerate(CLASSES):
            self.sample(str(cid), label)
        self.assertEqual([r['objects'][0]['class_id'] for r in self.report()['records']], list(range(6)))

    def test_rectangle_and_hand_calculation(self):
        self.sample()
        report = self.report()
        self.assertTrue(report['ok'])
        obj = report['records'][0]['objects'][0]
        self.assertEqual(obj['xyxy'], [20, 10, 100, 60])
        self.assertEqual(to_yolo(obj['xyxy'], 200, 100), (0.3, 0.35, 0.4, 0.5))
        self.assertFalse(report['partition_verified'])

    def test_round_trip_reverse_drag_and_edges(self):
        for points, expected in [([[100, 60], [20, 10]], [20, 10, 100, 60]),
                                 ([[0, 0], [200, 100]], [0, 0, 200, 100]),
                                 ([[0.125, 1.25], [199.75, 99.5]], [0.125, 1.25, 199.75, 99.5])]:
            with self.subTest(points=points):
                self.sample(points=points)
                box = self.report()['records'][0]['objects'][0]['xyxy']
                self.assertEqual(box, expected)
                reconstructed = from_yolo(to_yolo(box, 200, 100), 200, 100)
                self.assertLess(max(abs(a-b) for a,b in zip(box, reconstructed)), 1e-9)

    def test_bad_coordinates(self):
        for points, code in [([[1, 1], [1, 2]], 'ZERO_AREA'),
                              ([[-1, 0], [20, 20]], 'OUT_OF_BOUNDS'),
                              ([[0, 0], [201, 20]], 'OUT_OF_BOUNDS'),
                              ([[math.nan, 0], [20, 20]], 'NONFINITE'),
                              ([[0, 0], [math.inf, 20]], 'NONFINITE'),
                              ([[True, 0], [20, 20]], 'INVALID_COORDINATES')]:
            with self.subTest(points=points):
                self.sample(points=points)
                self.assert_error(code)

    def test_bad_inverse(self):
        for coords in [(0, 0, 0, 1), (1, 1, 1, 1), (math.nan, .5, .1, .1)]:
            with self.assertRaises(DataError):
                from_yolo(coords, 200, 100)

    def test_serialized_round_trip_1000_boxes(self):
        rng = random.Random(6)
        for index in range(1000):
            box = [0 if index % 2 else rng.random()*100,
                   0 if index % 3 else rng.random()*100,
                   641 if index % 5 else 100+rng.random()*500,
                   479 if index % 7 else 100+rng.random()*300]
            values = [float(format(v, '.12g')) for v in to_yolo(box, 641, 479)]
            restored = from_yolo(values, 641, 479)
            self.assertLess(max(abs(a-b) for a,b in zip(box,restored)), 1e-7)

    def test_empty_export_and_source_unchanged(self):
        path = self.sample(shapes=[])
        original = path.read_bytes()
        out = self.root / 'out'
        export_dataset(self.report(confirmed_empty={'a.png'}), out)
        self.assertEqual((out / 'labels/unpartitioned/a.txt').read_bytes(), b'')
        self.assertEqual(path.read_bytes(), original)
        self.assertFalse(json.loads((out / 'conversion.json').read_text(encoding='utf-8'))['partition_verified'])

    def test_refuse_nested_output_and_report_source_overwrite(self):
        path = self.sample()
        original = path.read_bytes()
        with self.assertRaises(DataError):
            export_dataset(self.report(), self.input / 'out')
        self.assertEqual(self.cli('check','--input',self.input,'--report',path).returncode, 2)
        self.assertEqual(path.read_bytes(), original)

    def test_invalid_schema_and_exif(self):
        for updates in [dict(shapes=None), dict(imageWidth=True), dict(flags=[]),
                        dict(shapes=[None]), dict(shapes=[dict(label=['bad'])])]:
            self.sample(**updates)
            self.assertFalse(self.report()['ok'])
        self.sample()
        exif = Image.Exif()
        exif[274] = 6
        Image.new('RGB', (200,100)).save(self.input / 'a.png', exif=exif)
        self.assert_error('EXIF_ORIENTATION')

    def test_missing_image_and_annotation(self):
        self.sample()
        (self.input / 'a.png').unlink()
        self.assert_error('MISSING_IMAGE')
        self.sample()
        (self.input / 'a.json').unlink()
        self.assert_error('MISSING_JSON')

    def test_unknown_label_shape_dimensions(self):
        self.sample(label='screw')
        self.assert_error('UNKNOWN_LABEL')
        self.sample(shapes=[dict(label='long_screw', shape_type='polygon', points=[[0, 0],[1,1],[2,3]])])
        self.assert_error('UNSUPPORTED_SHAPE')
        self.sample(imageWidth=100)
        self.assert_error('DIMENSION_MISMATCH')

    def test_empty_requires_explicit_confirmation(self):
        self.sample(shapes=[])
        self.assert_error('UNCONFIRMED_EMPTY')
        report = self.report(confirmed_empty={'a.png'})
        self.assertTrue(report['ok'])
        self.assertEqual(report['records'][0]['objects'], [])
        self.assertNotIn('predicted_label', report['records'][0])

    def test_unagreed_metadata_and_duplicates(self):
        self.sample(annotation_status='REVIEW')
        self.assert_error('UNAGREED_STATUS')
        self.sample(flags={'ignore': True})
        self.assert_error('UNAGREED_FLAGS')
        shape = dict(label='long_screw', shape_type='rectangle', points=[[20,10],[100,60]])
        self.sample(shapes=[dict(shape, ignore=True)])
        self.assert_error('UNAGREED_IGNORE')
        self.sample(shapes=[shape, shape])
        self.assert_error('DUPLICATE_OBJECT')
        self.sample(shapes=[shape, dict(shape, label='hex_nut')])
        self.assert_error('CONFLICTING_OBJECT')

    def test_malformed_json_paths_and_image(self):
        path = self.sample()
        path.write_text('{', encoding='utf-8')
        self.assert_error('INVALID_JSON')
        self.sample(imagePath='../a.png')
        self.assert_error('UNSAFE_IMAGE_PATH')
        self.sample()
        (self.input / 'a.png').write_bytes(b'not an image')
        self.assert_error('INVALID_IMAGE')

    def test_scene_leakage(self):
        self.sample('a'); self.sample('bb')
        self.assert_error('SCENE_LEAKAGE', manifest=self.manifest([
            ['a.png','scene1','train'], ['bb.png','scene1','val']]))

    def test_duplicate_image_id_and_content_leakage(self):
        self.sample('a'); self.sample('bb')
        manifest = self.manifest([['a.png','s1','train'], ['a.png','s2','val']])
        self.assert_error('DUPLICATE_IMAGE_ID', manifest=manifest)
        (self.input / 'bb.png').write_bytes((self.input / 'a.png').read_bytes())
        manifest = self.manifest([['a.png','s1','train'], ['bb.png','s2','val']])
        self.assert_error('IMAGE_LEAKAGE', manifest=manifest)

    def test_incomplete_partition(self):
        self.sample()
        for row in [['a.png','','train'], ['a.png','s1',''], ['a.png','s1','validation']]:
            self.assert_error('INVALID_PARTITION', manifest=self.manifest([row]))
        self.assert_error('MISSING_MANIFEST_ROW', manifest=self.manifest([]))

    def test_nested_input_and_ambiguous_stems(self):
        self.sample()
        Image.new('RGB',(200,100)).save(self.input / 'a.jpg')
        self.assert_error('AMBIGUOUS_STEM')
        (self.input / 'a.jpg').unlink()
        nested = self.input / 'nested'
        nested.mkdir()
        (nested / 'b.json').write_text('{}')
        self.assert_error('NESTED_INPUT')

    def test_manifest_extra_and_missing_columns(self):
        self.sample()
        self.assert_error('MISSING_MANIFEST_IMAGE', manifest=self.manifest([
            ['a.png','s1','train'], ['missing.png','s2','test']]))
        path = self.manifest([])
        path.write_text('image_id,split\na.png,train\n')
        self.assert_error('INVALID_MANIFEST', manifest=path)

    def test_cli_invalid_does_not_create_output(self):
        self.sample(label='bad')
        out = self.root / 'blocked'
        self.assertEqual(self.cli('convert','--input',self.input,'--output',out).returncode, 1)
        self.assertFalse(out.exists())

    def test_cli_write_failure_does_not_publish_partial_output(self):
        from tools.p06_prepare import main
        self.sample()
        out = self.root / 'out'
        real_write = Path.write_text

        def fail_objects(path, *args, **kwargs):
            if path.name == 'objects.json':
                raise OSError('injected disk failure after label write')
            return real_write(path, *args, **kwargs)

        with patch.object(Path, 'write_text', fail_objects):
            self.assertEqual(main(['convert', '--input', str(self.input), '--output', str(out)]), 2)
        self.assertFalse(out.exists(), 'Failed conversion must not publish a partial directory')
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ['input'])

    def test_test_isolation_and_output_contents(self):
        self.sample('a'); self.sample('bb', 'non_catalogue')
        manifest = self.manifest([['a.png','s1','train'], ['bb.png','s2','test']])
        report = self.report(manifest=manifest)
        self.assertTrue(report['partition_verified'])
        out = self.root / 'out'
        export_dataset(report, out)
        self.assertEqual((out / 'labels/train/a.txt').read_text(encoding='utf-8').strip(), '1 0.3 0.35 0.4 0.5')
        self.assertFalse(list(out.rglob('bb.txt')))
        saved = json.loads((out / 'objects.json').read_text(encoding='utf-8'))
        self.assertEqual([r['image_id'] for r in saved], ['a.png'])
        self.assertEqual(saved[0]['objects'][0]['xyxy'], [20,10,100,60])
        with self.assertRaises(FileExistsError):
            export_dataset(report, out)

    def test_invalid_report_cannot_export(self):
        self.sample(label='bad')
        out = self.root / 'out'
        with self.assertRaises(DataError):
            export_dataset(self.report(), out)
        self.assertFalse(out.exists())

    def cli(self, *args):
        return subprocess.run([sys.executable, '-B', 'tools/p06_prepare.py', *map(str,args)],
                              capture_output=True, text=True, encoding='utf-8')

    def test_cli_help_success_failure_and_no_overwrite(self):
        self.assertEqual(self.cli('--help').returncode, 0)
        self.sample()
        self.assertEqual(self.cli('check','--input',self.input).returncode, 0)
        out = self.root / 'out'
        result = self.cli('convert','--input',self.input,'--output',out)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.cli('convert','--input',self.input,'--output',out).returncode, 2)
        self.sample(label='bad')
        report = self.root / 'errors.json'
        self.assertEqual(self.cli('check','--input',self.input,'--report',report).returncode, 1)
        self.assertEqual(json.loads(report.read_text(encoding='utf-8'))['errors'][0]['type'], 'UNKNOWN_LABEL')
        self.assertEqual(self.cli('check','--input',self.input,'--report',report).returncode, 2)


if __name__ == '__main__':
    unittest.main()
