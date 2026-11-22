"""P04 baseline1: HOG + scikit-learn MLP image-level classifier.

Package layout
--------------
catalogue  : fixed five-class catalogue plus the reserved non_catalogue label 0.
features   : image loading and HOG feature extraction.
dataset    : ground-truth CSV loading and sample building.
model      : MLP bundle train / save / load / predict.
evaluate   : image-level metrics and live-latency measurement.
"""

from . import catalogue, features, dataset, model, evaluate  # noqa: F401

MODEL_ID = "baseline1"
