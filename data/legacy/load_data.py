import warnings
warnings.simplefilter('ignore')
import tensorflow_datasets as tfds
import tensorflow as tf
import time
import numpy as np

from config.hyper_parameters import *

start_time = time.time()
print("Load data start.")

examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True, as_supervised=True)
train_examples = examples['train']
val_examples = examples['validation']

"""
TFDS를 사용해서 TED talks open translation project의 포르투갈-영어 번역 데이터셋을 불러 오겠습니다
이 데이터셋은 5만개에 가까운 학습 데이터와 1100개의 검증 데이터, 2000개의 테스트 데이터를 가지고 있습니다.
학습 데이터셋에서 사용자 정의 서브워드 토크나이저를 생성합니다.
"""
tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
    (en.numpy() for pt, en in train_examples), target_vocab_size=2 ** 13)

tokenizer_pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
    (pt.numpy() for pt, en in train_examples), target_vocab_size=2 ** 13)

sample_string = 'Transformer is awesome.'

tokenized_string = tokenizer_en.encode(sample_string)
print('토큰화된 문자열은 {} 입니다'.format(tokenized_string))

original_string = tokenizer_en.decode(tokenized_string)
print('원래 문자열: {}'.format(original_string))

assert original_string == sample_string
for ts in tokenized_string:
    print('{} ----> {}'.format(ts, tokenizer_en.decode([ts])))


def encode(source, target):
    source = [tokenizer_pt.vocab_size] + tokenizer_pt.encode(source.numpy()) + [tokenizer_pt.vocab_size + 1]
    target = [tokenizer_en.vocab_size] + tokenizer_en.encode(target.numpy()) + [tokenizer_en.vocab_size + 1]

    return source, target


def tf_encode(pt, en):
    result_pt, result_en = tf.py_function(encode, [pt, en], [tf.int64, tf.int64])
    result_pt.set_shape([None])
    result_en.set_shape([None])

    return result_pt, result_en


def filter_max_length(x, y, max_length=MAX_SEQ_LEN):
    return tf.logical_and(tf.size(x) <= max_length,
                          tf.size(y) <= max_length)


train_dataset = train_examples.map(tf_encode)
train_dataset = train_dataset.filter(filter_max_length)

train_dataset = train_dataset.cache()
train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE, ((None,), (None,)))
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
valid_dataset = val_examples.map(tf_encode)
valid_dataset = valid_dataset.filter(filter_max_length).padded_batch(BATCH_SIZE, ((None,), (None,)))

encoder_vocab_size = tokenizer_pt.vocab_size + 2
decoder_vocab_size = tokenizer_en.vocab_size + 2
