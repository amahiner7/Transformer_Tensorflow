import os
from datetime import datetime

BASE_DATASET_DIR = "./datasets/UTKFace"
ORIGINAL_DATA_FILE_BASE = os.path.join(BASE_DATASET_DIR, "Original")
RIFINE_DATA_FILE_BASE = os.path.join(BASE_DATASET_DIR, "RefineData")
CSV_FILE_PATH = os.path.join(BASE_DATASET_DIR, "face.csv")
REFINE_DATA_DIR = os.path.join(BASE_DATASET_DIR, "RefineData")
UTKFACE_DATA_DIR = os.path.join(BASE_DATASET_DIR, "Data")
LOG_ROOT_DIR = "./log"
TRAINING_LOG_DIR = os.path.join(LOG_ROOT_DIR, "training")
DATETIME_DIR = datetime.now().strftime("%Y%m%d-%H%M%S")
TENSORBOARD_LOG_DIR = os.path.join(TRAINING_LOG_DIR, DATETIME_DIR)
TENSORBOARD_LEARNING_RATE_LOG_DIR = os.path.join(TENSORBOARD_LOG_DIR, "learning_rate")
MODEL_FILE_ROOT_DIR = "./model_files"
MODEL_FILE_DIR = os.path.join(MODEL_FILE_ROOT_DIR, DATETIME_DIR)
MODEL_FILE_PATH = os.path.join(MODEL_FILE_DIR, "ResNet50_UTKFace_epoch_{epoch:02d}_val_loss_{val_loss:.4f}.h5")

if not os.path.exists(LOG_ROOT_DIR):
    os.mkdir(LOG_ROOT_DIR)

if not os.path.exists(TRAINING_LOG_DIR):
    os.mkdir(TRAINING_LOG_DIR)

if not os.path.exists(MODEL_FILE_ROOT_DIR):
    os.mkdir(MODEL_FILE_ROOT_DIR)

if not os.path.exists(MODEL_FILE_DIR):
    os.mkdir(MODEL_FILE_DIR)
