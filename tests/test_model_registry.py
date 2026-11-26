import hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from src import model_registry as registry
from src.p07_dashboard import regions

class RegistryTests(unittest.TestCase):
    def test_gui_exposes_exactly_the_three_frozen_baselines(self):
        models=[name.split(':')[1] for name,_ in regions() if name.startswith('model:')]
        self.assertEqual(models,list(registry.PRODUCTION_MODELS))
    def test_final_is_not_an_alias_for_baseline3(self):
        with self.assertRaisesRegex(RuntimeError,'NOT_READY'):registry.resolve('final',{})
    def test_hash_mismatch_is_error_and_registry_overrides_stale_model_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);weight=root/'frozen.pt';weight.write_bytes(b'frozen')
            entry=dict(files={'frozen.pt':hashlib.sha256(b'frozen').hexdigest()},model_config={'bundle':'approved'},
                bridge='actual:factory',scene_guard=False,version='frozen')
            config=dict(schema='frozen-model-registry/v1',models={m:entry for m in registry.PRODUCTION_MODELS},shared_files={})
            (root/'registry.json').write_text(json.dumps(config))
            with patch.object(registry,'ROOT',root):
                resolved,files=registry.resolve('baseline1',dict(model_registry='registry.json',model_config={'bundle':'stale'},_live_scene_guard=True))
                self.assertEqual(resolved['model_config']['bundle'],'approved');self.assertFalse(resolved['_live_scene_guard'])
                weight.write_bytes(b'corrupt')
                with self.assertRaisesRegex(RuntimeError,'SHA256 mismatch'):registry.resolve('baseline1',dict(model_registry='registry.json'))
