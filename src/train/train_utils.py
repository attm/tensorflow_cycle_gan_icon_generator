import numpy as np
from random import randint, random
from os.path import join as pjoin
import os
import tensorflow as tf
import json


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

def save_json(data : object, save_path : str) -> None:
    """
    Saves data in json format.

    Parameters:
        data (object) : data to be saved.
        save_path (str) : data will be saved in that file.
    Returns:
        None
    """
    with open(save_path, "w") as f:
        json.dump(data, f)

def load_json(load_path : str) -> object:
    """
    Loads json from path. 

    Parameters:
        load_path (str) : json file with data that need to be loaded.
    Returns:
        loaded (object) : loaded file from json.
    """
    if not os.path.exists(load_path):
        raise FileNotFoundError("load_json: load_path not exists, got path {0}".format(load_path))

    with open(load_path, "r") as f:
        return json.load(f)

def load_train_logs(logs_folder_path : str) -> list:
    """
    Loads logs from folder.

    Parameters:
        logs_folder_path (str) : folder path from which logs will be loaded.
    Returns:
        logs (list) : list of logs, order is gen_AtoB_losses, gen_BtoA_losses, dis_A_losses, dis_B_losses
    """
    if not os.path.exists(logs_folder_path):
        raise FileNotFoundError("load_train_logs: logs_folder_path not exists, got path {0}".format(logs_folder_path))

    gen_AtoB_losses = load_json(pjoin(logs_folder_path, "gen_AtoB_losses.json"))
    gen_BtoA_losses = load_json(pjoin(logs_folder_path, "gen_BtoA_losses.json"))
    dis_A_losses = load_json(pjoin(logs_folder_path, "dis_A_losses.json"))
    dis_B_losses = load_json(pjoin(logs_folder_path, "dis_B_losses.json"))
    return [gen_AtoB_losses, gen_BtoA_losses, dis_A_losses, dis_B_losses]

def save_train_logs(logs : list, logs_folder_path : str) -> None:
    """
    Saves logs to the given folder, logs will be saved in json format.

    Parameters:
        logs (list) : logs to be saved, list must contain gen_AtoB_losses, gen_BtoA_losses, dis_A_losses, dis_B_losses
        logs_folder_path (str) : path of the folder where logs will be saved.
    Returns:
        None
    """
    if not os.path.exists(logs_folder_path):
        raise FileNotFoundError("load_train_logs: logs_folder_path not exists, got path {0}".format(logs_folder_path))

    gen_AtoB_losses, gen_BtoA_losses, dis_A_losses, dis_B_losses = logs
    save_json(gen_AtoB_losses, pjoin(logs_folder_path, "gen_AtoB_losses.json"))
    save_json(gen_BtoA_losses, pjoin(logs_folder_path, "gen_BtoA_losses.json"))
    save_json(dis_A_losses, pjoin(logs_folder_path, "dis_A_losses.json"))
    save_json(dis_B_losses, pjoin(logs_folder_path, "dis_B_losses.json"))
    return None
