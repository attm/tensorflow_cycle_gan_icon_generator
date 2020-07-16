import pytest
from PIL.Image import Image
from os.path import join as pjoin
import os
from src.data_process.image_process import open_image, apply_blur, apply_palette_reduction, process_image


cwd = os.path.dirname(os.path.realpath(__file__))
TEST_IMG_PATH = pjoin(cwd, "test_data", "test_img.png")


def test_open_image():
    img = open_image(TEST_IMG_PATH)
    assert isinstance(img, Image)

def test_apply_blur():
    img = open_image(TEST_IMG_PATH)
    img = apply_blur(img, blur_radius=5)
    assert isinstance(img, Image)

def test_apply_palette_reduction():
    img = open_image(TEST_IMG_PATH)
    img = apply_palette_reduction(img, reduced_palette_colors_count=5)
    assert isinstance(img, Image)

def test_process_image():
    img = open_image(TEST_IMG_PATH)
    img = process_image(img)
    assert isinstance(img, Image)