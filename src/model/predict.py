import tensorflow as tf
import numpy as np
import os


class CycleGanPredictor():
    def __init__(self, model : tf.keras.Model, predicted_save_folder : str = None):
        """
        Initializes new instance of CycleGanPredictor. 

        Parameters:
            model (tf.keras.Model) : model that will be used for predictioning.
            predicted_save_folder (str) : folder, where all predicted images will be saved.
        """
        if isinstance(model, tf.keras.Model):
            self.model = model
        else:
            raise TypeError("CycleGanPredictor.__init__() : expected model of type tf.keras.Model, got {0}".format(type(model)))

        if isinstance(predicted_save_folder, str) or predicted_save_folder is None:
            self.predicted_save_folder = predicted_save_folder
        else:
            raise TypeError("CycleGanPredictor.__init__() : expected predicted_save_folder of type str, got {0}".format(type(predicted_save_folder)))

    def predict(self, image : np.ndarray) -> np.ndarray:
        """
        Transforms single image.

        Parameters:
            image (np.ndarray) : image to be transformed.
        Returns:
            transformed_image (np.ndarray) : image transformed by predictor model.
        """
        if isinstance(image, np.ndarray):
            transformed_image = self.predict_numpy(image)
            return transformed_image

        # Raise exception if no method found for this type of image
        raise TypeError("CycleGanPredictor.predict(): image type is not supported, supported types are : np.ndarray")

    def predict_numpy(self, image : np.ndarray) -> np.ndarray:
        """
        Predicts np.ndarray image.

        Parameters:
            image (np.ndarray) : image to be predicted.
        Returns:
            transformed_image (np.ndarray) : image transformed by predictor model.
        """
        transformed = self.model.predict(image)
        return transformed
