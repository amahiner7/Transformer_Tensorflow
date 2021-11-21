import os
from datetime import datetime


class FilePath(object):
    LOG_ROOT_DIR = "./log"
    TRAINING_LOG_DIR = os.path.join(LOG_ROOT_DIR, "training")
    DATETIME_DIR = datetime.now().strftime("%Y%m%d-%H%M%S")
    TENSORBOARD_LOG_DIR = os.path.join(TRAINING_LOG_DIR, DATETIME_DIR)
    TENSORBOARD_LEARNING_RATE_LOG_DIR = os.path.join(TENSORBOARD_LOG_DIR, "learning_rate")
    MODEL_FILE_ROOT_DIR = "./model_files"
    MODEL_FILE_DIR = os.path.join(MODEL_FILE_ROOT_DIR, DATETIME_DIR)
    MODEL_FILE_PATH = os.path.join(MODEL_FILE_DIR, "Transformer_epoch_{:02d}_val_loss_{:.4f}.h5")
    MODEL_FILE_PATH_FIT_FORM = os.path.join(MODEL_FILE_DIR, "Transformer_epoch_{epoch:02d}_val_loss_{val_loss:.4f}.h5")


def make_directories():
    if not os.path.exists(FilePath.LOG_ROOT_DIR):
        os.mkdir(FilePath.LOG_ROOT_DIR)

    if not os.path.exists(FilePath.TRAINING_LOG_DIR):
        os.mkdir(FilePath.TRAINING_LOG_DIR)

    if not os.path.exists(FilePath.MODEL_FILE_ROOT_DIR):
        os.mkdir(FilePath.MODEL_FILE_ROOT_DIR)

    if not os.path.exists(FilePath.MODEL_FILE_DIR):
        os.mkdir(FilePath.MODEL_FILE_DIR)
