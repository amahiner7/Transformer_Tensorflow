import warnings

warnings.simplefilter('ignore')

import tensorflow_datasets as tfds
import time
from config.hyper_parameters import HyperParameter
from data.legacy.DataLoader import DataLoader


def get_data_sample(data, sample_ratio):
    data_len = len(data)
    sample_count = int(data_len * sample_ratio)

    return data[:sample_count]


def load_data(sample_ratio=1.0):
    start_time = time.time()
    print("==================== Load data start. ====================")

    examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True, as_supervised=True)
    train_examples = examples['train']
    valid_examples = examples['validation']

    if sample_ratio < 1.0:
        train_examples = train_examples.take(int(len(train_examples) * sample_ratio))
        valid_examples = valid_examples.take(int(len(valid_examples) * sample_ratio))

    train_data_loader = DataLoader(dataset=train_examples,
                                   buffer_size=HyperParameter.BUFFER_SIZE,
                                   batch_size=HyperParameter.BATCH_SIZE,
                                   max_seq_len=HyperParameter.MAX_SEQ_LEN,
                                   is_train_data=True)

    valid_data_loader = DataLoader(dataset=valid_examples,
                                   buffer_size=HyperParameter.BUFFER_SIZE,
                                   batch_size=HyperParameter.BATCH_SIZE,
                                   max_seq_len=HyperParameter.MAX_SEQ_LEN,
                                   is_train_data=False)

    print("train_data_loader.dataset length: ", len(train_data_loader.dataset))
    print("valid_data_loader.dataset length: ", len(valid_data_loader.dataset))

    elapsed_time = time.time() - start_time
    print("==================== Load data complete.({:.1f} second) ====================".format(elapsed_time))

    return train_data_loader, valid_data_loader
