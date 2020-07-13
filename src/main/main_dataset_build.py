from os.path import join as pjoin
import os
from src.data_processing.dataset_build import process_images_folder, build_numpy_dataset_from_folder
 

cwd = os.getcwd()
DATA_FOLDER_PATH = pjoin(cwd, "data")
RAW_IMAGES_FOLDER_PATH = pjoin(DATA_FOLDER_PATH, "raw_images")
PROCESSED_IMAGES_FOLDER_PATH = pjoin(DATA_FOLDER_PATH, "processed_images")
DATASETS_FOLDER_PATH = pjoin(DATA_FOLDER_PATH, "datasets")

REBUILD_IMAGES = False
REBUILD_DATASETS = True

INPUT_DATASET_NAME = "input.npy"
TARGET_DATASET_NAME = "target.npy"

def rebuild_images():
    process_images_folder(RAW_IMAGES_FOLDER_PATH, PROCESSED_IMAGES_FOLDER_PATH)

def rebuild_datasets():
    input_dataset_file_path = pjoin(DATASETS_FOLDER_PATH, INPUT_DATASET_NAME)
    build_numpy_dataset_from_folder(PROCESSED_IMAGES_FOLDER_PATH, input_dataset_file_path, save_dtype="float32")

    target_dataset_file_path = pjoin(DATASETS_FOLDER_PATH, TARGET_DATASET_NAME)
    build_numpy_dataset_from_folder(RAW_IMAGES_FOLDER_PATH, target_dataset_file_path, save_dtype="float32")

def main():
    if REBUILD_IMAGES:
        rebuild_images()

    if REBUILD_DATASETS:
        rebuild_datasets()

if __name__ == "__main__":
    main()