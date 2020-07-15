import numpy as np
from random import randint, random
from os.path import join as pjoin
import os
import tensorflow as tf


def update_image_pool(pool : list, images : list, max_size : int = 50) -> np.ndarray:
    """
    Utility function for updating generated images pool randomly.

    Parameters:
        pool (list) : pool that will be updated.
        images (list) : list of images that will be added to the pool.
        max_size (int) : maximum size of the pool.
    Returns:
        selected (np.ndarray) : array of randovly selected images, size is max_size.
    """
    selected = []
    for image in images:
        if len(pool) < max_size:
            # stock the pool
            pool.append(image)
            selected.append(image)
        elif random() < 0.5:
            # use image, but don't add it to the pool
            selected.append(image)
        else:
            # replace an existing image and use replaced image
            ix = randint(0, len(pool)-1)
            selected.append(pool[ix])
            pool[ix] = image
    return np.array(selected)

def list_average(input_list : list) -> float:
    """
    Returns average of numbers in list.

    Parameters:
        input_list (list) : list of numbers.
    Returns:
        avg (float) : average value.
    """
    summ_value = 0.0
    for v in input_list:
        if isinstance(v, int) or isinstance(v, float):
            summ_value += v
    return summ_value / len(input_list)

def save_cyclegan_model(models : list, save_folder_path : str) -> None:
    """
    Saves cycleGAN model weight to the folder.

    Parameters:
        models (list) : list of models, order in list is d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA
        save_folder_path (str) : path of folder where models should be saved.
    Returns:
        None
    """
    if not os.path.exists(save_folder_path):
        raise FileNotFoundError("save_cyclegan_model: path not exists, got path {0}".format(save_folder_path))

    d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA = models
    
    d_model_A.save(pjoin(save_folder_path, "disA"))
    d_model_B.save(pjoin(save_folder_path, "disB"))
    g_model_AtoB.save(pjoin(save_folder_path, "genAtoB"))
    g_model_BtoA.save(pjoin(save_folder_path, "genBtoA"))
    c_model_AtoB.save(pjoin(save_folder_path, "compAtoB"))
    c_model_BtoA.save(pjoin(save_folder_path, "compBtoA"))

def load_cyclegan_model(load_folder_path : str) -> list:
    """
    Loads models from folder, returns list of models.

    Parameters:
        load_folder_path (str) : path of the folder where models are saved.
    Returns:
        models (list) : list of models, order is d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA
    """
    if not os.path.exists(load_folder_path):
        raise FileNotFoundError("load_cyclegan_model: load_folder_path not extists, got path {0}".format(load_folder_path))

    d_model_A = tf.keras.models.load_model(pjoin(load_folder_path, "disA"))
    d_model_B = tf.keras.models.load_model(pjoin(load_folder_path, "disB"))
    g_model_AtoB = tf.keras.models.load_model(pjoin(load_folder_path, "genAtoB"))
    g_model_BtoA = tf.keras.models.load_model(pjoin(load_folder_path, "genBtoA"))
    c_model_AtoB = tf.keras.models.load_model(pjoin(load_folder_path, "compAtoB"))
    c_model_BtoA = tf.keras.models.load_model(pjoin(load_folder_path, "compBtoA"))

    return [d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA]