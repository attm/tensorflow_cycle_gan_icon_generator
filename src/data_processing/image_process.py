from PIL import Image, ImageFilter
import PIL
from os import listdir
from os.path import isfile
from os.path import join as pjoin
from os.path import exists
import os


BLUR_blur_radius = 2
REDUCED_PALETTE_COLORS_COUNT = 3


def open_image(img_path : str) -> PIL.Image.Image:
    """
    Opens and returns PIL.Image.

    Parameters:
        img_path (str) : path of the image.
    Returns: 
        img (Image) : PIL.Image object.
    """
    if exists(img_path):
        img = Image.open(img_path, mode="r")
        return img
    else:
        raise FileNotFoundError("Path not exists, given path is {0}".format(img_path))

def save_image(img : PIL.Image.Image, img_path : str) -> None:
    """
    Saves PIL.Image.

    Parameters:
        img (PIL.Image) : image to save.
        img_path (str) : path of the image.
    Returns: 
        None.
    """
    img.save(img_path)

def apply_blur(img : PIL.Image.Image, blur_radius : int = BLUR_blur_radius) -> PIL.Image.Image:
    """
    Applies blur for PIL.Image image.

    Parameters:
        img (PIL.Image) : image to be blurred.
        blur_radius (int) : blur blur_radius, or blur strength.
    Returns: 
        blurred_image (PIL.Image) : blurred image.
    """
    if not isinstance(img, PIL.Image.Image):
        raise TypeError("apply_blur: expected img of type PIL.Image, got {0}".format(type(img)))

    blurred_image = img.convert("RGBA")
    blurred_image = blurred_image.filter(ImageFilter.GaussianBlur(blur_radius))
    return blurred_image

def apply_palette_reduction(img : PIL.Image.Image, reduced_palette_colors_count : int = REDUCED_PALETTE_COLORS_COUNT) -> PIL.Image.Image:
    """
    Applies palette reduction for PIL.Image image.

    Parameters:
        img (PIL.Image) : image to be processed.
        reduced_palette_colors_count (int) : count of colors for processed image.
    Returns: 
        processed_image (PIL.Image) : processed image with pallete count reduced.
    """
    if not isinstance(img, PIL.Image.Image):
        raise TypeError("apply_palette_reduction: expected img of type PIL.Image, got {0}".format(type(img)))

    processed_image = img.convert('P', palette=Image.ADAPTIVE, colors=reduced_palette_colors_count)
    return processed_image

def process_image(img : PIL.Image.Image, blur_radius : int = 2, reduced_palette_colors_count : int = 3) -> PIL.Image.Image:
    """
    Processing image, applying different processors.

    Parameters:
        img (PIL.Image) : image to be processed.
        reduced_palette_colors_count (int) : count of colors for processed image.
        blur_radius (int) : blur blur_radius, or blur strength.
    Returns: 
        processed_image (PIL.Image) : processed image.
    """
    processed_image = apply_blur(img, blur_radius=blur_radius)
    processed_image = apply_palette_reduction(processed_image, reduced_palette_colors_count=reduced_palette_colors_count)
    return processed_image

