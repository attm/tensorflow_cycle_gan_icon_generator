import numpy as np
from tensorflow.keras import Model
from random import random, randint
from src.models_building.cycle_gan_build import define_composite_model
from src.models_building.discriminator_build import define_discriminator
from src.models_building.generator_build import define_generator


def build_cycle_gan(image_shape : tuple) -> list:
    """
    Builds default cycle gan model

    Parameters:
        image_shape (tuple of shape height x width x chanels) : shape of image that will be used in datasets.
    Returns:
        models (list) : list of cycle gan models, order is d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA
    """
    g_model_AtoB = define_generator(image_shape)
    # generator: B -> A
    g_model_BtoA = define_generator(image_shape)
    # discriminator: A -> [real/fake]
    d_model_A = define_discriminator(image_shape)
    # discriminator: B -> [real/fake]
    d_model_B = define_discriminator(image_shape)
    # composite: A -> B -> [real/fake, A]
    c_model_AtoB = define_composite_model(g_model_AtoB, d_model_B, g_model_BtoA, image_shape)
    # composite: B -> A -> [real/fake, B]
    c_model_BtoA = define_composite_model(g_model_BtoA, d_model_A, g_model_AtoB, image_shape)
    models = [d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA]
    return models

def generate_patch_labels(batch_size : int, patch_shape : int, label : int = 1) -> np.ndarray:
    """
    Generates labels for patchgan.

    Parameters:
        batch_size (int) : how many samples is needed.
        patch_shape (int) : height and width of patch.
        label (int) : label int, usually 0 for fake and 1 for real.
    Returns:
        labels (np.ndarray) : labels array, shape is (batch_size, patch_shape, patch_shape, 1).
    """
    labels = np.full(shape=(batch_size, patch_shape, patch_shape, 1), fill_value=label)
    return labels

def generate_fake_samples(generator_model : Model, dataset : np.ndarray, patch_shape : int) -> np.ndarray:
    """
    Generates X, y pairs where labels are fake (0) and X are generated with given generator.

    Parameters:
        generator_model (keras Model) : generator model that will be used for making prediction.
        dataset (np.ndarray) : dataset that will be input for generator.
        patch_shape (int) : height & width of shape.
    Returns:
        X (np.ndarray) : generated images.
        y (np.ndarray) : array of labels.
    """
    X = generator_model.predict(dataset)
    y = generate_patch_labels(len(X), patch_shape, label=0)
    return X, y

def generate_real_samples(dataset : np.ndarray, patch_shape : int) -> np.ndarray:
    """
    Generates X, y pairs from dataset, where labels are real (1).

    Parameters:
        dataset (np.ndarray) : dataset that will be used as X.
        patch_shape (int) : height & width of shape.
    Returns:
        X (np.ndarray) : just a dataset (for usability).
        y (np.ndarray) : real labels.
    """
    y = generate_patch_labels(len(dataset), patch_shape, label=1)
    return dataset, y
