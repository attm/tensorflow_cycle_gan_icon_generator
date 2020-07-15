import pytest
from src.train.train_utils import list_average


def test_list_average():
    test_list = [0, -1, 10, 50.5, None, "string", False]
    assert list_average(test_list) == 8.5
