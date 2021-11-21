import tensorflow as tf
import tensorflow_datasets as tfds


class DataLoader:
    def __init__(self, dataset, buffer_size, batch_size, max_seq_len, is_train_data=True):
        self.dataset = dataset
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (en.numpy() for pt, en in dataset), target_vocab_size=2 ** 13)
        self.tokenizer_pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, en in dataset), target_vocab_size=2 ** 13)
        self.encoder_vocab_size = self.tokenizer_pt.vocab_size + 2
        self.decoder_vocab_size = self.tokenizer_en.vocab_size + 2

        if is_train_data:
            self.item = dataset.map(self.tf_encode)
            self.item = self.item.filter(self.filter_max_length)
            self.item = self.item.cache()
            self.item = self.item.shuffle(buffer_size).padded_batch(batch_size, ((None,), (None,)))
            self.item = self.item.prefetch(tf.data.experimental.AUTOTUNE)
        else:
            self.item = dataset.map(self.tf_encode)
            self.item = self.item.filter(self.filter_max_length).padded_batch(batch_size, ((None,), (None,)))

    def __len__(self):
        return self.item.reduce(0, lambda x, _: x + 1).numpy()

    def item_length(self):
        item_length = 0
        for source, _ in self.item:
            item_length += len(source)

        return item_length

    def encode(self, source, target):
        source = [self.tokenizer_pt.vocab_size] + self.tokenizer_pt.encode(source.numpy()) + \
                 [self.tokenizer_pt.vocab_size + 1]
        target = [self.tokenizer_en.vocab_size] + self.tokenizer_en.encode(target.numpy()) + \
                 [self.tokenizer_en.vocab_size + 1]

        return source, target

    def tf_encode(self, pt, en):
        result_pt, result_en = tf.py_function(self.encode, [pt, en], [tf.int64, tf.int64])
        result_pt.set_shape([None])
        result_en.set_shape([None])

        return result_pt, result_en

    def filter_max_length(self, x, y):
        return tf.logical_and(tf.size(x) <= self.max_seq_len, tf.size(y) <= self.max_seq_len)
