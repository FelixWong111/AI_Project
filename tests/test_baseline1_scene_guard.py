import tempfile
import unittest
from pathlib import Path
import numpy as np
from src.models.baseline1.scene_guard import SceneGuard


class SceneGuardTests(unittest.TestCase):
    def test_support_check_and_checkpoint_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'guard.npz'
            np.savez(path,reference=np.zeros((262,1764)),limit=1.,model_sha256='checkpoint')
            guard=SceneGuard(path,'checkpoint')
            self.assertTrue(guard.check(np.zeros((1,1764)))[0])
            self.assertFalse(guard.check(np.ones((1,1764)))[0])
            with self.assertRaisesRegex(ValueError,'another'):
                SceneGuard(path,'wrong-checkpoint')
