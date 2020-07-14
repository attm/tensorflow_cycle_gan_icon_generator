import pytest
from src.models_building.model_utils import generate_patch_labels


def test_generate_patch_labels():
    labels = generate_patch_labels(batch_size=4, patch_shape=8, label=0)
    assert labels.shape == (4, 8, 8, 1)