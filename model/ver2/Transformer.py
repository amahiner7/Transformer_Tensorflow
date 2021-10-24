from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import time
import math

from model.ver1.layers.Encoder import Encoder
from model.ver1.layers.Decoder import Decoder
from config.hyper_parameters import *
from utils.common import *
from utils.CustomSchedule import CustomSchedule

train_step_signature = [tf.TensorSpec(shape=(None, None), dtype=tf.int64),
                        tf.TensorSpec(shape=(None, None), dtype=tf.int64)]


class Transformer(Model):
    def __init__(self, d_input, d_output, d_embed, d_model, d_ff, num_heads, num_layers,
                 dropout_prob, seq_len, name='Transformer_ver2_Custom_training_loop'):
        super().__init__(name=name)

        self.encoder = Encoder(d_input=d_input,
                               d_embed=d_embed,
                               d_model=d_model,
                               d_ff=d_ff,
                               num_layers=num_layers,
                               num_heads=num_heads,
                               seq_len=seq_len,
                               dropout_prob=dropout_prob)

        self.decoder = Decoder(d_output=d_output,
                               d_embed=d_embed,
                               d_model=d_model,
                               d_ff=d_ff,
                               num_layers=num_layers,
                               num_heads=num_heads,
                               seq_len=seq_len,
                               dropout_prob=dropout_prob)

        self.learning_rate_schedule = CustomSchedule(d_model)
        self.optimizer = Adam(learning_rate=self.learning_rate_schedule, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        self.training = True

        self.train_metric_loss = tf.keras.metrics.Mean(name='train_metric_loss')
        self.train_metric_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_metric_accuracy')
        self.valid_metric_loss = tf.keras.metrics.Mean(name='valid_metric_loss')
        self.valid_metric_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='valid_metric_accuracy')

    def criterion(self, label, pred):
        loss_object = SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        mask = tf.math.logical_not(tf.math.equal(label, 0))
        loss_ = loss_object(label, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)

    def make_source_mask(self, source):
        """
        :param source: shape (batch_size, source_length)
        :return source_mask: shape(batch_size, 1, 1, source_length)
        """
        source_mask = tf.cast(tf.math.equal(source, 0), tf.float32)

        return source_mask[:, tf.newaxis, tf.newaxis, :]

    def make_target_mask(self, target):
        """
        target: shape (batch_size, target_length)
        """
        target_len = tf.shape(target)[1]
        target_sub_mask = 1 - tf.linalg.band_part(tf.ones((target_len, target_len)), -1, 0)
        padding_mask = self.make_source_mask(target)

        return tf.maximum(target_sub_mask, padding_mask)

    def call(self, inputs):
        """
        source shape: (batch_size, source_len)
        target shape: (batch_size, target_len)
        """
        source, target = inputs

        # source_mask shape: (batch_size, 1, 1, source_len)
        # target_mask shape: (batch_size, 1, target_len, target_len)
        source_mask = self.make_source_mask(source=source)
        target_mask = self.make_target_mask(target=target)

        # encoder_output shape: (batch_size, source_len, model_dim)
        encoder_output = self.encoder(source=source, mask=source_mask, training=self.training)

        # output shape: (batch_size, target_len, output_dim)
        # attention shape: (batch_size, num_heads, target_len, src_len)
        output, attention = self.decoder(decoder_source=target, decoder_mask=target_mask,
                                         encoder_source=encoder_output, encoder_mask=source_mask,
                                         training=self.training)

        return output, attention

    @tf.function(input_signature=train_step_signature)
    def _tf_train_on_batch(self, source, target):
        target_input = target[:, :-1]
        target_real = target[:, 1:]

        with tf.GradientTape() as tape:
            predictions, _ = self.call((source, target_input))
            loss = self.criterion(target_real, predictions)

        gradients = tape.gradient(loss, self.trainable_variables)

        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.train_metric_loss(loss)
        self.train_metric_accuracy(target_real, predictions)

    @tf.function(input_signature=train_step_signature)
    def _tf_evaluate_on_batch(self, source, target):
        target_input = target[:, :-1]
        target_real = target[:, 1:]

        predictions, _ = self.call((source, target_input))
        loss = self.criterion(target_real, predictions)

        self.valid_metric_loss(loss)
        self.valid_metric_accuracy(target_real, predictions)

    def train_on_batch(self, data_loader, log_interval):
        for batch_index, (source, target) in enumerate(data_loader.item):
            self._tf_train_on_batch(source=source, target=target)

            if batch_index % log_interval == 0 and batch_index is not 0:
                print(" BATCH: [{}/{}({:.0f}%)] | TRAIN LOSS: {:.4f}, ACCURACY: {:.4f}".format(
                    batch_index * len(source),
                    len(data_loader.dataset),
                    100.0 * batch_index / len(data_loader),
                    self.train_metric_loss.result(),
                    self.train_metric_accuracy.result()))

    def evaluate_on_batch(self, data_loader):
        for batch_index, (source, target) in enumerate(data_loader.item):
            self._tf_evaluate_on_batch(source=source, target=target)

    def train_on_epoch(self, train_data, valid_data, epochs, log_interval=1):
        train_metric_loss_history = []
        train_metric_acc_history = []
        valid_metric_loss_history = []
        valid_metric_acc_history = []

        for epoch in range(epochs):
            print('=============== TRAINING EPOCHS {} / {} =============== '.format(epoch + 1, epochs))
            train_start_time = time.time()

            self.train_metric_loss.reset_states()
            self.train_metric_accuracy.reset_states()
            self.valid_metric_loss.reset_states()
            self.valid_metric_accuracy.reset_states()

            self.train_on_batch(data_loader=train_data, log_interval=log_interval)
            self.evaluate_on_batch(data_loader=valid_data)

            print("TRAIN LOSS: {:.4f}, ACC: {:.2f}, PPL: {:.4f} | VALID LOSS: {:.4f}, ACC: {:.2f}, PPL: {:.4f} | "
                  "ELAPSED TIME: {}\n".
                  format(self.train_metric_loss.result(),
                         self.train_metric_accuracy.result() * 100.0,
                         math.exp(self.train_metric_loss.result()),
                         self.valid_metric_loss.result(),
                         self.valid_metric_accuracy.result() * 100.0,
                         math.exp(self.valid_metric_loss.result()),
                         format_time(time.time() - train_start_time)))

            train_metric_loss_history.append(self.train_metric_loss.result())
            train_metric_acc_history.append(self.train_metric_accuracy.result())
            valid_metric_loss_history.append(self.valid_metric_loss.result())
            valid_metric_acc_history.append(self.valid_metric_accuracy.result())

        history = {'loss': train_metric_loss_history, 'accuracy': train_metric_acc_history,
                   'val_loss': valid_metric_loss_history, 'val_accuracy': valid_metric_acc_history}

        return history

    def evaluate_sentence(self, sentence, tokenizer_pt, tokenizer_en, max_seq_len):
        self.training = False
        start_token = [tokenizer_pt.vocab_size]
        end_token = [tokenizer_pt.vocab_size + 1]

        # 입력 문장은 포르투갈어이므로 start 토큰과 end 토큰을 추가합니다.
        sentence = start_token + tokenizer_pt.encode(sentence) + end_token
        encoder_input = tf.expand_dims(sentence, 0)

        # 타겟은 영어이므로 트랜스포머의 첫번째 단어는 영어 start 토큰입니다.
        decoder_input = [tokenizer_en.vocab_size]
        output = tf.expand_dims(decoder_input, 0)

        for i in range(max_seq_len):
            # predictions.shape == (batch_size, seq_len, vocab_size)
            predictions, attention_weights = self.call((encoder_input, output))

            # seq_len 차원에서 마지막 단어를 선택합니다.
            predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

            # predicted_id가 end 토큰과 같으면 결과를 반환합니다.
            if predicted_id == tokenizer_en.vocab_size + 1:
                return tf.squeeze(output, axis=0), attention_weights

            # predicted_id를 디코더의 입력값으로 들어가는 출력값과 연결합니다.
            output = tf.concat([output, predicted_id], axis=-1)

        return tf.squeeze(output, axis=0), attention_weights

    def translate(self, sentence, tokenizer_en):
        result, attention_weights = self.evaluate_sentence(sentence)

        predicted_sentence = tokenizer_en.decode([i for i in result
                                                  if i < tokenizer_en.vocab_size])

        print('Input: {}'.format(sentence))
        print('Predicted translation: {}'.format(predicted_sentence))

    def build_graph(self, encoder_input_shape, decoder_input_size, batch_size):
        encoder_input = Input(shape=encoder_input_shape, batch_size=batch_size)
        decoder_input = Input(shape=decoder_input_size, batch_size=batch_size)
        return Model(inputs=[encoder_input, decoder_input], outputs=self.call((encoder_input, decoder_input)))

    def summary_model(self):
        temp_source = tf.random.uniform((BATCH_SIZE, 38), dtype=tf.int64, minval=0, maxval=200)
        temp_target = tf.random.uniform((BATCH_SIZE, 36), dtype=tf.int64, minval=0, maxval=200)
        encoder_input_shape = temp_source.shape[-1]
        decoder_input_size = temp_target.shape[-1]
        batch_size = temp_target.shape[0]

        temp_model = self.build_graph(encoder_input_shape=encoder_input_shape,
                                      decoder_input_size=decoder_input_size,
                                      batch_size=batch_size)
        temp_model.summary()
