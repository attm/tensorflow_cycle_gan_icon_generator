from os.path import join as pjoin
import os
import numpy as np
import time
import tensorflow as tf
from src.models_building.model_utils import generate_fake_samples, generate_patch_labels, update_image_pool, generate_real_samples, build_cycle_gan


# Datasets path
cwd = os.getcwd()
DATASETS_FOLDER_PATH = pjoin(cwd, "data", "datasets")
INPUT_DATASET_NAME = pjoin(DATASETS_FOLDER_PATH, "input.npy")
TARGET_DATASET_NAME = pjoin(DATASETS_FOLDER_PATH, "target.npy")

# Model parameters
IMG_SHAPE = (60, 60, 3)

# Training parameters
BATCH_SIZE = 4
N_EPOCHS = 500
PATCH_SHAPE = 4
USE_CPU = False

if USE_CPU:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
else:
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_virtual_device_configuration(physical_devices[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5000)])

tf.keras.backend.clear_session()

def load_data():
    input_dataset = np.load(INPUT_DATASET_NAME)
    target_dataset = np.load(TARGET_DATASET_NAME)
    return input_dataset, target_dataset

def train_cycle_gan_on_batch(models : list, pools : list, input_batch : np.ndarray, target_batch : np.ndarray, patch_shape : int = 8) -> None:
    # A is input dataset, B is target
    # Unpacking models
    start_time = time.time()
    d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA = models
    pool_A, pool_B = pools

    # Generating real images X-y pairs
    X_real_A, y_real_A = generate_real_samples(input_batch, patch_shape=patch_shape)
    X_real_B, y_real_B = generate_real_samples(target_batch, patch_shape=patch_shape)

    # Generationg fake images X-y pairs using generators model 
    X_fake_A, y_fake_A = generate_fake_samples(g_model_BtoA, X_real_B, patch_shape=patch_shape)
    X_fake_B, y_fake_B = generate_fake_samples(g_model_AtoB, X_real_A, patch_shape=patch_shape)

    # Updating fake image pools
    X_fake_A = update_image_pool(pool_A, X_fake_A)
    X_fake_B = update_image_pool(pool_B, X_fake_B)

    # Training B to A generator
    g_model_BtoA_loss, _, _, _, _  = c_model_BtoA.train_on_batch([X_real_B, X_real_A], [y_real_A, X_real_A, X_real_B, X_real_A])

    # Training discriminator A on real and fake images
    dis_A_loss1 = d_model_A.train_on_batch(X_real_A, y_real_A)
    dis_A_loss2 = d_model_A.train_on_batch(X_fake_A, y_fake_A)

    # Training A to B generator
    g_model_AtoB_loss, _, _, _, _ = c_model_AtoB.train_on_batch([X_real_A, X_real_B], [y_real_B, X_real_B, X_real_A, X_real_B])

    # Training discriminator B on real and fake images
    dis_B_loss1 = d_model_B.train_on_batch(X_real_B, y_real_B)
    dis_B_loss2 = d_model_B.train_on_batch(X_fake_B, y_fake_B)
    
    # Checking end time of training
    end_time = time.time()

    print("\nCycle GAN trained on batch in {:.2f} seconds".format(end_time-start_time))
    print("\nGenerators losses are:")
    print("    Generator AtoB (input to target) loss is {0}".format(g_model_AtoB_loss))
    print("    Generator BtoA (target to input) loss is {0}".format(g_model_BtoA_loss))
    print("Discriminators losses are:")
    print("    Discriminator A (input) losses are {0} and {1}".format(dis_A_loss1, dis_A_loss2))
    print("    Discriminator B (target) losses are {0} and {1}".format(dis_B_loss1, dis_B_loss2))

def main():
    # Loading training data
    input_data, target_data = load_data()
    print("Loaded datasets:")
    print("Input dataset shape is {0}".format(input_data.shape))
    print("Target dataset shape is {0}".format(target_data.shape))

    # Building models
    models = build_cycle_gan(IMG_SHAPE)
    # Defining pools
    pool_A = []
    pool_B = []
    pools = [pool_A, pool_B]

    # MAIN TRAINING CYCLE
    for i in range(N_EPOCHS):
        # Getting batch of data
        input_batch = input_data[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        target_batch = target_data[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        # Training cyclegan
        print("############################################################")
        print("EPOCHS: {0}/{1}".format(i, N_EPOCHS))
        train_cycle_gan_on_batch(models, pools, input_batch, target_batch, patch_shape=PATCH_SHAPE)

if __name__ == "__main__":
    main()