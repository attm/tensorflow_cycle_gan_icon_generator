import tensorflow as tf
import numpy as np
from os.path import join as pjoin
import os
from src.model.predict import CycleGanPredictor


physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_virtual_device_configuration(physical_devices[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5000)])
tf.keras.backend.clear_session()

cwd = os.getcwd()
INPUT_DATASET_PATH = pjoin(cwd, "data", "datasets", "input.npy")
PREDICTOR_MODEL_PATH = pjoin(cwd, "models", "genAtoB")
TRANSFORMED_IMAGES_FOLDER_PATH = pjoin(cwd, "data", "transformed")


def main():
    input_dataset = np.load(INPUT_DATASET_PATH)
    data_to_predict = input_dataset[5].reshape(1, 60, 60, 3)

    predictor_model = tf.keras.models.load_model(PREDICTOR_MODEL_PATH)

    predictor = CycleGanPredictor(predictor_model)

    transformed = predictor.predict(data_to_predict)
    print(transformed.shape)
    np.save(pjoin(TRANSFORMED_IMAGES_FOLDER_PATH, "transformed.npy"), transformed)

if __name__ == "__main__":
    main()