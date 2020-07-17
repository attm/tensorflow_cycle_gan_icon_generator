from os.path import join as pjoin
import os
import numpy as np
import time
import tensorflow as tf
from src.models_build.model_utils import generate_fake_samples, generate_patch_labels, generate_real_samples, build_cycle_gan
from src.train.train_utils import update_image_pool, list_average, save_cyclegan_model, load_cyclegan_model, save_train_logs, load_train_logs


# Datasets path
cwd = os.getcwd()
DATASETS_FOLDER_PATH = pjoin(cwd, "data", "datasets")
INPUT_DATASET_NAME = pjoin(DATASETS_FOLDER_PATH, "input.npy")
TARGET_DATASET_NAME = pjoin(DATASETS_FOLDER_PATH, "target.npy")
SAVED_MODELS_PATH = pjoin(cwd, "models")
LOGS_FOLDER_PATH = pjoin(cwd, "logs")

# Model parameters
IMG_SHAPE = (60, 60, 3)

# Training parameters
BATCH_SIZE = 4
N_EPOCHS = 1001
PATCH_SHAPE = 4
USE_CPU = False
LOAD_SAVED_MODEL = True
SAVE_LOGS = True
LOAD_LOGS = True
SAVE_MODEL_N_EPOCHS_EACH = 1000

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
    return [g_model_AtoB_loss, g_model_BtoA_loss, dis_A_loss1, dis_A_loss2, dis_B_loss1, dis_B_loss2]

def main():
    # Loading training data
    input_data, target_data = load_data()
    print("Loaded datasets:")
    print("Input dataset shape is {0}".format(input_data.shape))
    print("Target dataset shape is {0}".format(target_data.shape))

    # Building models (or loading if LOAD_SAVED_MODEL is True)
    if LOAD_SAVED_MODEL:
        try:
            models = load_cyclegan_model(SAVED_MODELS_PATH)
            print("Loaded models from folder")
        except Exception:
            models = build_cycle_gan(IMG_SHAPE)
            print("Folder not found or models not exists, building new")
    else:
        models = build_cycle_gan(IMG_SHAPE)
        print("Built models")
    # Defining pools
    pool_A = []
    pool_B = []
    pools = [pool_A, pool_B]

    # Generator losses
    gen_AtoB_losses_avg = []
    gen_BtoA_losses_avg = []
    gen_AtoB_losses = []
    gen_BtoA_losses = []
    # Discriminator losses
    dis_A_losses_avg = []
    dis_B_losses_avg = []
    dis_A_losses = []
    dis_B_losses = []

    # Defining training logs lists
    if LOAD_LOGS:
        try:
            logs = load_train_logs(LOGS_FOLDER_PATH)
            gen_AtoB_losses, gen_BtoA_losses, dis_A_losses, dis_B_losses = logs
            print("Loaded logs from {0}".format(LOGS_FOLDER_PATH))
        except Exception:
            print("Can't load logs from {0}".format(LOGS_FOLDER_PATH))

    # MAIN TRAINING CYCLE
    for i in range(N_EPOCHS):
        # Getting batch of data
        input_batch = input_data[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        target_batch = target_data[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        # Training cyclegan
        print("############################################################")
        print("EPOCHS: {0}/{1}".format(i, N_EPOCHS))
        logs = train_cycle_gan_on_batch(models, pools, input_batch, target_batch, patch_shape=PATCH_SHAPE)
        g_model_AtoB_loss, g_model_BtoA_loss, dis_A_loss1, dis_A_loss2, dis_B_loss1, dis_B_loss2 = logs

        # Calctulating losses averages
        # AtoB Generator losses
        gen_AtoB_losses.append(g_model_AtoB_loss)
        gen_AtoB_loss_avg = list_average(gen_AtoB_losses)
        gen_AtoB_losses_avg.append(gen_AtoB_loss_avg)
        # BtoA Generator losses
        gen_BtoA_losses.append(g_model_BtoA_loss)
        gen_BtoA_loss_avg = list_average(gen_BtoA_losses)
        gen_BtoA_losses_avg.append(gen_BtoA_loss_avg)
        # Discriminator A losses
        dis_A_loss = (dis_A_loss1 + dis_A_loss2) / 2.0
        dis_A_losses.append(dis_A_loss)
        dis_A_loss_avg = list_average(dis_A_losses)
        dis_A_losses_avg.append(dis_A_loss_avg)
        # Discriminator B losses
        dis_B_loss = (dis_B_loss1 + dis_B_loss2) / 2.0
        dis_B_losses.append(dis_B_loss)
        dis_B_loss_avg = list_average(dis_B_losses)
        dis_B_losses_avg.append(dis_B_loss_avg)

        print("\nGenerators losses are:")
        print("    Generator AtoB (input to target) loss is {0}".format(gen_AtoB_loss_avg))
        print("    Generator BtoA (target to input) loss is {0}".format(gen_BtoA_loss_avg))
        print("Discriminators losses are:")
        print("    Discriminator A (input) loss is {0}".format(dis_A_loss_avg))
        print("    Discriminator B (target) loss is {0}".format(dis_B_loss_avg))

        # Saving model each number of epochs
        if i % SAVE_MODEL_N_EPOCHS_EACH == 0  and i >= SAVE_MODEL_N_EPOCHS_EACH:
            save_cyclegan_model(models, SAVED_MODELS_PATH)
            print("Models saved at epoch number {0}".format(i))

        # Saving logs each epoch
        if SAVE_LOGS:
            logs = [gen_AtoB_losses, gen_BtoA_losses, dis_A_losses, dis_B_losses]
            save_train_logs(logs, LOGS_FOLDER_PATH)

if __name__ == "__main__":
    main()