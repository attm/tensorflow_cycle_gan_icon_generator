import pytest
import os
from os.path import join as pjoin
from src.data_process.dataset_build import build_numpy_dataset_from_folder


cwd = os.path.dirname(os.path.realpath(__file__))
TEST_IMG_FOLDER_PATH = pjoin(cwd, "test_data")


def test_build_numpy_dataset_from_folder(tmpdir):
    save_dir = tmpdir.mkdir("dataset_dir")
    npdataset = build_numpy_dataset_from_folder(TEST_IMG_FOLDER_PATH)
    assert npdataset.shape == (1, 60, 60, 3)