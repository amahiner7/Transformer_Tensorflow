import warnings

warnings.simplefilter('ignore')

import tensorflow_datasets as tfds
import tensorflow as tf
import time
from config.hyper_parameters import *
from data.legacy.DataLoader import DataLoader

start_time = time.time()
print("==================== Load data start. ====================")

examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True, as_supervised=True)
train_examples = examples['train']
valid_examples = examples['validation']

"""
TFDS를 사용해서 TED talks open translation project의 포르투갈-영어 번역 데이터셋을 불러 오겠습니다
이 데이터셋은 5만개에 가까운 학습 데이터와 1100개의 검증 데이터, 2000개의 테스트 데이터를 가지고 있습니다.
학습 데이터셋에서 사용자 정의 서브워드 토크나이저를 생성합니다.
"""


train_data_loader = DataLoader(dataset=train_examples,
                               buffer_size=BUFFER_SIZE,
                               batch_size=BATCH_SIZE,
                               max_seq_len=MAX_SEQ_LEN,
                               is_train_data=True)

valid_data_loader = DataLoader(dataset=valid_examples,
                               buffer_size=BUFFER_SIZE,
                               batch_size=BATCH_SIZE,
                               max_seq_len=MAX_SEQ_LEN,
                               is_train_data=False)

print("train_data_loader length: ", len(train_data_loader))
print("train_data_loader.dataset length: ", len(train_data_loader.dataset))

print("valid_examples length: ", len(valid_data_loader))
print("valid_examples.dataset length: ", len(valid_data_loader.dataset))

elapsed_time = time.time() - start_time
print("==================== Load data complete.({:.1f} second) ====================".format(elapsed_time))
