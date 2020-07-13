from os.path import join as pjoin
from os.path import exists
import os
import PIL
import numpy as np
from src.data_processing.image_process import process_image, open_image, save_image


cwd = os.getcwd()
DATA_FOLDER_PATH = pjoin(cwd, "data")
RAW_IMAGES_FOLDER_PATH = pjoin(DATA_FOLDER_PATH, "raw_images")
PROCESSED_IMAGES_FOLDER_PATH = pjoin(DATA_FOLDER_PATH, "processed_images")
DATASETS_FOLDER_PATH = pjoin(DATA_FOLDER_PATH, "datasets")

DEFAULT_PROCESS_FUNCTION = process_image


def process_images_folder(raw_images_folder_path : str, processed_images_folder_path : str, process_function : callable = DEFAULT_PROCESS_FUNCTION) -> None:
    """
    Processing all images from folder. 

    Parameters:
        raw_images_folder_path (str) : folder of raw images to be processd.
        processed_images_folder_path (str) : folder where all processed images need to be saved.
        process_function (callable) : function used for processing each image,
                                      should have first argument of type PIL.Image,
                                      should return image of type PIL.Image
    Returns:
        None
    """
    if not exists(raw_images_folder_path):
        raise AttributeError("process_images_folder: raw_images_folder_path not exists, given is {0}".format(raw_images_folder_path))

    if not exists(processed_images_folder_path):
        raise AttributeError("process_images_folder: processed_images_folder_path not exists, given is {0}".format(processed_images_folder_path))

    files_list = os.listdir(raw_images_folder_path)
    
    # Processing all images in raw_images folder
    for file_name in files_list:
        # Defining load & save image path
        img_path = pjoin(raw_images_folder_path, file_name)
        saved_img_path = pjoin(processed_images_folder_path, file_name)
        # Processing image and saving it
        img = open_image(img_path)
        processed_image = process_function(img)
        save_image(processed_image, saved_img_path)

def build_numpy_dataset_from_folder(folder_path : str, dataset_save_path : str = None, rescale : int = 1.0/255.0) -> np.ndarray:
    """
    Builds numpy ndarray from images. Optionally save it.

    Parameters:
        folder_path (str) : folder path with images.
        dataset_save_path (str) : path for dataset saving.
        rescale (int) : each pixel of image will be multiplied by rescale value.
    Returns:
        dataset (np.ndarray) : dataset built from folder.
    """
    if not exists(folder_path):
        raise FileNotFoundError("build_numpy_dataset_from_folder: given folder_path not exists, given {0}".format(folder_path))
    
    files_list = os.listdir(folder_path)
    dataset_list = []
    for file_name in files_list:
        img = open_image(pjoin(folder_path, file_name))
        np_img = ndarray_from_img(img, rescale=rescale)
        dataset_list.append(np_img)
    
    dataset = np.array(dataset_list)

    if isinstance(dataset_save_path, str) and dataset_save_path is not False:
        np.save(dataset_save_path, dataset)
    return dataset
    
def ndarray_from_img(img : PIL.Image.Image, rescale: int = 1.0/255.0) -> np.ndarray:
    """
    Making np.ndarray from PIL Image.

    Parameters:
        img (PIL Image) : image to be transformed into numpy ndarray.
    Returns:
        ndarray_image (np.ndarray) : array, format is height x width x channels.
    """
    ndarray_image = np.array(img)
    ndarray_image = ndarray_image * rescale
    return ndarray_image
    