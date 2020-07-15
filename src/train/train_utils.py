import numpy as np
from random import randint, random


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

