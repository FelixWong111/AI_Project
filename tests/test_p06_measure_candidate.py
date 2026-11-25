"""Candidate metrology checks: synthetic fixtures, no Train/Val/Test access."""
from pathlib import Path
import hashlib
import tempfile
import unittest

import cv2
import numpy as np
from PIL import Image

from tools.p06_measure_candidate import (
    candidate_contour, detect_ticks, fit_ticks, generate, load_display,
    metric_envelope, regular_run,
)


class MeasureCandidateTests(unittest.TestCase):
    def test_known_scale_with_alternating_holdout(self):
        result = fit_ticks(17 + 25 * np.arange(41, dtype=float))
        self.assertAlmostEqual(result['px_per_mm'], 25)
        self.assertEqual(result['fit_count'], 21)
        self.assertEqual(result['holdout_count'], 20)
        self.assertLess(result['holdout_max_abs_mm'], 1e-10)
        moved = 17 + 25 * np.arange(41, dtype=float)
        moved[1::2] += 2
        self.assertAlmostEqual(fit_ticks(moved)['holdout_max_abs_mm'], .08)

    def test_invalid_and_missing_ticks_are_not_silently_renumbered(self):
        for points in ([1, 1, 2], [3, 2, 1], [0, float('nan'), 2],
                       [0, float('inf'), 2], np.arange(10),
                       np.r_[np.arange(20), np.arange(21, 41)]):
            with self.subTest(points=str(points)), self.assertRaises(ValueError):
                fit_ticks(points)
        pts = np.r_[np.arange(25) * 30, 900 + np.arange(35) * 30]
        np.testing.assert_array_equal(regular_run(pts), pts[25:])

    def test_detect_horizontal_and_vertical_tick_strokes(self):
        horizontal = np.full((100, 500), 255, np.uint8)
        vertical = np.full((500, 100), 255, np.uint8)
        for index in range(21):
            cv2.line(horizontal, (20 + index * 20, 10), (20 + index * 20, 80), 0, 3)
            cv2.line(vertical, (10, 20 + index * 20), (80, 20 + index * 20), 0, 3)
        for axis, gray, roi in [('x', horizontal, (0, 0, 500, 100)),
                                ('y', vertical, (0, 0, 100, 500))]:
            detected = detect_ticks(gray, roi, axis)
            np.testing.assert_allclose(detected, 20 + 20 * np.arange(21), atol=.1)
            self.assertAlmostEqual(fit_ticks(detected)['px_per_mm'], 20)

    def test_exif6_and_no_exif_preserve_source_and_known_pixel_position(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'fixture.png'
            image = Image.new('RGB', (60, 40), 'white')
            image.putpixel((10, 20), (255, 0, 0))
            exif = Image.Exif(); exif[274] = 6
            image.save(path, exif=exif)
            before = hashlib.sha256(path.read_bytes()).hexdigest()
            display, meta = load_display(path)
            self.assertEqual(display.size, (40, 60))
            self.assertEqual(display.getpixel((19, 10)), (255, 0, 0))
            self.assertEqual(meta['original_width'], 60)
            self.assertEqual(meta['exif_orientation'], 6)
            self.assertEqual(before, hashlib.sha256(path.read_bytes()).hexdigest())
            image.save(path)
            self.assertEqual(load_display(path)[0].size, (60, 40))

    def test_metric_envelope_anisotropic_scale(self):
        contour = np.array([[[100, 100]], [[300, 100]], [[300, 200]], [[100, 200]]], np.int32)
        result = metric_envelope(contour, 20, 10)
        self.assertAlmostEqual(result['long_mm_candidate'], 10)
        self.assertAlmostEqual(result['short_mm_candidate'], 10)
        with self.assertRaises(ValueError):
            metric_envelope(contour, 0, 10)

    def test_empty_multiple_and_boundary_contours(self):
        image = np.full((400, 500), 255, np.uint8)
        roi = (50, 50, 450, 350)
        self.assertIsNone(candidate_contour(image, roi))
        cv2.rectangle(image, (100, 100), (200, 200), 0, -1)
        contour = candidate_contour(image, roi)
        self.assertEqual(cv2.boundingRect(contour), (100, 100, 101, 101))
        cv2.rectangle(image, (250, 100), (350, 200), 0, -1)
        with self.assertRaisesRegex(ValueError, 'MULTIPLE'):
            candidate_contour(image, roi)
        image[:] = 255
        cv2.rectangle(image, (40, 100), (200, 200), 0, -1)
        with self.assertRaisesRegex(ValueError, 'BOUNDARY'):
            candidate_contour(image, roi)

    def test_existing_output_and_source_overlap_are_rejected_before_writes(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / 'source'; source.mkdir()
            output = Path(temp) / 'output'; output.mkdir()
            sentinel = output / 'keep.txt'; sentinel.write_text('keep')
            with self.assertRaises(FileExistsError):
                generate(source, output)
            with self.assertRaises(ValueError):
                generate(source, source / 'derived')
            self.assertEqual(sentinel.read_text(encoding='utf-8'), 'keep')
            self.assertEqual(list(source.iterdir()), [])

    def test_bad_input_does_not_publish_a_partial_experiment(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / 'source'; source.mkdir()
            for index in range(1, 7):
                Image.new('RGB', (80, 60), 'white').save(source / f'{index}.jpg')
            hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()}
            output = Path(temp) / 'derived'
            with self.assertRaisesRegex(ValueError, '3024x4032'):
                generate(source, output)
            self.assertFalse(output.exists())
            self.assertEqual(hashes, {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()})


if __name__ == '__main__':
    unittest.main()
