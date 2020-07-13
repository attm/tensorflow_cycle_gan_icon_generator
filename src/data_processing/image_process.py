from PIL import Image, ImageFilter
from os import listdir
from os.path import isfile
from os.path import join as pjoin
from os.path import exists
import os


def open_image(img_path : str) -> Image:
    """
    Opens and returns PIL.Image.

    Parameters:
        img_path (str) : path of the image.
    Returns: 
        img (Image) : PIL.Image object.
    """
    if exists(img_path):
        img = Image.open(img_path)
        return img
    else:
        raise FileExistsError("Path not exists, given path is {0}".format(img_path))

def save_image(img : Image, img_path : str) -> None:
    """
    Saves PIL.Image.

    Parameters:
        img (PIL.Image) : image to save.
        img_path (str) : path of the image.
    Returns: 
        None.
    """
    img.save(img_path)

def apply_blur(img : Image, radius : int = 2) -> Image:
    """
    Applies blur for PIL.Image image.

    Parameters:
        img (PIL.Image) : image to be blurred.
        radius (int) : blur radius, or blur strength.
    Returns: 
        blurred_image (PIL.Image) : blurred image.
    """
    if not isinstance(img, Image):
        raise TypeError("apply_blur: expected img of type PIL.Image, got {0}".format(type(img)))

    blurred_image = img.convert("RGB")
    blurred_image = img.filter(ImageFilter.GaussianBlur(radius))
    return blurred_image

def apply_palette_reduction(img, colors_count=3):
    """
    Applies palette reduction for PIL.Image image.

    Parameters:
        img (PIL.Image) : image to be processed.
        colors_count (int) : count of colors for processed image.
    Returns: 
        processed_image (PIL.Image) : processed image with pallete count reduced.
    """
    if not isinstance(img, Image):
        raise TypeError("apply_palette_reduction: expected img of type PIL.Image, got {0}".format(type(img)))

    processed_image = img.convert('P', palette=Image.ADAPTIVE, colors=colors_count)
    return processed_image