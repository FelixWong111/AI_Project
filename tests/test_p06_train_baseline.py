"""Guard the fixed training entry point without starting a model."""
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import io
from contextlib import redirect_stderr
from unittest.mock import patch

from tools import p06_train_baseline as baseline


class BaselineTests(unittest.TestCase):
    def test_no_mode_cannot_start_default_long_training(self):
        with patch.object(sys,'argv',['entry']),redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as stopped: baseline.main()
            self.assertEqual(stopped.exception.code,2)

    def test_test_path_refused_before_environment_or_data_read(self):
        with TemporaryDirectory() as temporary,patch.object(baseline,'inspect_environment') as environment, \
             patch('tools.p06_accepted_data.accepted_plan') as data:
            with self.assertRaisesRegex(ValueError,'Test'):
                baseline.preflight(baseline.ROOT/'data/p03/test_v2',baseline.WEIGHTS,Path(temporary)/'new')
            environment.assert_not_called();data.assert_not_called()

    def test_check_data_never_calls_train(self):
        run=baseline.RUN.parent/'unit_check_data_nonexistent'
        with patch.object(sys,'argv',['entry','--check-data','--output-dir',str(run)]), \
             patch.object(baseline,'preflight',return_value=({},{})),patch.object(baseline,'snapshot',return_value={}), \
             patch.object(baseline,'configure_runtime'),patch('tools.p06_accepted_data.prepare_runtime',return_value={'status':'PASS'}), \
             patch.object(baseline,'train') as training:
            self.assertEqual(baseline.main(),0);training.assert_not_called();self.assertFalse(run.exists())

    def test_yaml_rejects_mapping_and_test_path(self):
        config = dict(path=".", train="images/train", val="images/val",
                      nc=6, names=dict(enumerate(baseline.CLASSES)))
        baseline.validate_yaml(config)
        with self.assertRaisesRegex(ValueError, "forbidden"):
            baseline.validate_yaml(dict(config, test="never-open"))
        config["names"][0] = "background"
        with self.assertRaisesRegex(ValueError, "mapping"):
            baseline.validate_yaml(config)

    def test_bad_path_and_existing_output(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "already exists"):
                baseline.preflight(baseline.DATA, baseline.WEIGHTS, root)
            with self.assertRaisesRegex(ValueError, "published development dataset"):
                baseline.preflight(root/"missing", baseline.WEIGHTS, root/"new")

    def test_cuda_unavailable_stops_preflight(self):
        torch = SimpleNamespace(version=SimpleNamespace(cuda="12.6"),
                                cuda=SimpleNamespace(is_available=lambda: False))
        with patch.object(sys,"prefix","ai-p06-yolo"), patch.object(sys,"version_info",(3,11,18)), \
             patch.object(baseline,"version",side_effect=baseline.VERSIONS.__getitem__):
            with self.assertRaisesRegex(ValueError, "CUDA 12.6 unavailable"):
                baseline.inspect_environment(torch)

    def test_prediction_diagnostic_tracks_confusion_and_duplicate(self):
        truth = [{"class_id": 1, "xyxy":[0,0,10,10]}, {"class_id": 2, "xyxy":[20,20,30,30]}]
        predictions = [dict(class_id=0,xyxy=truth[0]["xyxy"],confidence=.9),
                       dict(class_id=1,xyxy=truth[0]["xyxy"],confidence=.8)]
        result = baseline.compare_predictions(predictions,truth)
        self.assertEqual(result["confusions_gt_to_pred"],[[1,0]])
        self.assertEqual(result["unmatched_gt"],[2])
        self.assertEqual(result["unmatched_predictions"],[{"class_id":1,"possible_duplicate":True}])


    def test_smoke_changes_only_epochs(self):
        self.assertEqual(baseline.training_params(), baseline.PARAMS)
        self.assertEqual(baseline.training_params(True), dict(baseline.PARAMS, epochs=1))

    def test_snapshot_allows_only_official_cache_changes(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/"labels").mkdir()
            (root/"labels/train.txt").write_text("original")
            before = baseline.snapshot(root)
            for split in ("train", "val"):
                (root/f"labels/{split}.cache").write_bytes(b"regenerable")
            self.assertEqual(before, baseline.snapshot(root))
            (root/"labels/train.txt").write_text("changed")
            self.assertNotEqual(before, baseline.snapshot(root))
            (root/"other.cache").write_bytes(b"unexpected")
            self.assertIn("other.cache", baseline.snapshot(root))

    def test_dry_run_does_not_train_or_configure_or_create_output(self):
        with TemporaryDirectory() as temporary:
            run = baseline.RUN.parent/"test_dry_run_nonexistent"
            with patch.object(sys,"argv",["entry", "--dry-run", "--smoke", "--output-dir",str(run)]), \
                 patch.object(baseline,"preflight",return_value=({},{})) as checked, \
                 patch.object(baseline,"train") as train, \
                 patch.object(baseline,"configure_runtime") as configure:
                self.assertEqual(baseline.main(),0)
                checked.assert_called_once()
                train.assert_not_called()
                configure.assert_not_called()
                self.assertFalse(run.exists())

    def test_official_label_cache_create_reuse_and_invalid_version_rebuild(self):
        from PIL import Image
        import numpy as np
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/"config").mkdir()
            with patch.dict("os.environ", {"YOLO_CONFIG_DIR": str(root/"config")}):
                from ultralytics.data.dataset import YOLODataset
            images = root/"images/train"
            labels = root/"labels/train"
            images.mkdir(parents=True)
            labels.mkdir(parents=True)
            Image.new("RGB",(64,48),"white").save(images/"sample.png")
            (labels/"sample.txt").write_text("1 0.5 0.5 0.5 0.5\n")
            def load():
                return YOLODataset(img_path=str(images), imgsz=640, batch_size=8,
                                   augment=False, cache=False,
                                   data={"names":dict(enumerate(baseline.CLASSES)),"nc":6,"channels":3})
            self.assertEqual(len(load()),1)
            cache = root/"labels/train.cache"
            self.assertTrue(cache.is_file())
            initial = cache.read_bytes()
            self.assertEqual(len(load().labels[0]["cls"]),1)
            self.assertEqual(cache.read_bytes(),initial)
            stored = np.load(cache,allow_pickle=True).item()
            stored["version"] = "invalid"
            with cache.open("wb") as stream:
                np.save(stream,stored)
            self.assertEqual(len(load().labels[0]["cls"]),1)
            self.assertNotEqual(np.load(cache,allow_pickle=True).item()["version"],"invalid")
