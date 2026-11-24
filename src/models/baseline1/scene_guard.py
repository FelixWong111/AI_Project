"""Training-support check for the live system, separate from raw baseline scoring."""
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist


class SceneGuard:
    def __init__(self, path, model_sha256):
        with np.load(Path(path), allow_pickle=False) as data:
            if str(data['model_sha256']) != model_sha256:
                raise ValueError('Scene guard belongs to another baseline1 checkpoint')
            self.reference = data['reference']
            self.limit = float(data['limit'])
        if self.reference.shape != (262, 1764) or not np.isfinite(self.reference).all() or not self.limit > 0:
            raise ValueError('Invalid scene guard artifact')

    def check(self, standardized_feature):
        distance = float(cdist(standardized_feature, self.reference).min())
        return distance <= self.limit, distance
