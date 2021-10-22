from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import time
import math

from model.ver1.layers.Encoder import Encoder
from model.ver1.layers.Decoder import Decoder
from config.file_path import *
from config.hyper_parameters import *
from utils.common import *
from utils.CosineAnnealingWarmUpRestarts import CosineAnnealingWarmUpRestarts
from utils.LearningRateHistory import LearningRateHistory
from utils.CustomSchedule import CustomSchedule


train_step_signature = [tf.TensorSpec(shape=(None, None), dtype=tf.int64),
                        tf.TensorSpec(shape=(None, None), dtype=tf.int64)]


class Transformer(Model):
    def __init__(self, d_input, d_output, d_embed, d_model, d_ff, num_heads, num_layers,
                 dropout_prob, seq_len, name='Transformer'):
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
        # self.optimizer = Adam(learning_rate=self.learning_rate_schedule, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        self.optimizer = Adam(learning_rate=LEARNING_RATE)

        # self.train_metric_loss = tf.keras.metrics.Mean(name='train_metric_loss')
        # self.train_metric_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_metric_accuracy')
        # self.valid_metric_loss = tf.keras.metrics.Mean(name='valid_metric_loss')
        # self.valid_metric_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='valid_metric_accuracy')

    @tf.function()
    def train_step(self, data):
        # Unpack the data
        source, target = data
        target_input = target[:, :-1]
        target_real = target[:, 1:]

        with tf.GradientTape() as tape:
            # Compute predictions
            predictions, _ = self.call((source, target_input))

            # Updates the metrics tracking the loss
            loss = self.compiled_loss(target_real, predictions, regularization_losses=self.losses)

        # Compute gradients
        trainable_var = self.trainable_variables
        gradients = tape.gradient(loss, trainable_var)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_var))

        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(target_real, predictions)

        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def make_callbacks(self):
        callbacks = []

        model_check_point = ModelCheckpoint(filepath=MODEL_FILE_PATH,
                                            monitor='val_loss',
                                            save_weights_only=True,
                                            save_best_only=True,
                                            verbose=1)

        tensorboard = TensorBoard(log_dir=TENSORBOARD_LOG_DIR)

        learning_rate_scheduler = CosineAnnealingWarmUpRestarts(initial_learning_rate=LEARNING_RATE,
                                                                first_decay_steps=1,
                                                                alpha=0.0,
                                                                t_mul=2.0,
                                                                m_mul=1.0)
        learning_rate_history = LearningRateHistory(log_dir=TENSORBOARD_LEARNING_RATE_LOG_DIR)

        # callbacks.append(model_check_point)
        callbacks.append(tensorboard)
        callbacks.append(learning_rate_scheduler)
        callbacks.append(learning_rate_history)

        return callbacks

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
        encoder_output = self.encoder(source=source, mask=source_mask)

        # output shape: (batch_size, target_len, output_dim)
        # attention shape: (batch_size, num_heads, target_len, src_len)
        output, attention = self.decoder(decoder_source=target, decoder_mask=target_mask,
                                         encoder_source=encoder_output, encoder_mask=source_mask)

        return output, attention

    # @tf.function(input_signature=train_step_signature)
    # def _tf_train_on_batch(self, source, target):
    #     target_input = target[:, :-1]
    #     target_real = target[:, 1:]
    #
    #     with tf.GradientTape() as tape:
    #         predictions, _ = self.call(source, target_input)
    #         loss = self.criterion(target_real, predictions)
    #
    #     gradients = tape.gradient(loss, self.trainable_variables)
    #
    #     self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
    #
    #     self.train_metric_loss(loss)
    #     self.train_metric_accuracy(target_real, predictions)

    # @tf.function(input_signature=train_step_signature)
    # def _tf_evaluate_on_batch(self, source, target):
    #     target_input = target[:, :-1]
    #     target_real = target[:, 1:]
    #
    #     predictions, _ = self.call(source, target_input)
    #     loss = self.criterion(target_real, predictions)
    #
    #     self.valid_metric_loss(loss)
    #     self.valid_metric_accuracy(target_real, predictions)

    # def train_on_batch(self, data_loader, log_interval):
    #     for batch_index, (source, target) in enumerate(data_loader.item):
    #         self._tf_train_on_batch(source=source, target=target)
    #
    #         if batch_index % log_interval == 0 and batch_index is not 0:
    #             print(" BATCH: [{}/{}({:.0f}%)] | TRAIN LOSS: {:.4f}, ACCURACY: {:.4f}".format(
    #                 batch_index * len(source),
    #                 len(data_loader.dataset),
    #                 100.0 * batch_index / len(data_loader),
    #                 self.train_metric_loss.result(),
    #                 self.train_metric_accuracy.result()))
    #
    # def evaluate_on_batch(self, data_loader):
    #     for batch_index, (source, target) in enumerate(data_loader.item):
    #         self._tf_evaluate_on_batch(source=source, target=target)
    #
    # def train_on_epoch(self, train_data_loader, valid_data_loader, epochs, log_interval=1):
    #     for epoch in range(epochs):
    #         print('=============== TRAINING EPOCHS {} / {} =============== '.format(epoch + 1, epochs))
    #         train_start_time = time.time()
    #
    #         self.train_metric_loss.reset_states()
    #         self.train_metric_accuracy.reset_states()
    #         self.valid_metric_loss.reset_states()
    #         self.valid_metric_accuracy.reset_states()
    #
    #         self.train_on_batch(data_loader=train_data_loader, log_interval=log_interval)
    #         self.evaluate_on_batch(data_loader=valid_data_loader)
    #
    #         print("TRAIN LOSS: {:.4f}, ACC: {:.2f}, PPL: {:.4f} | VALID LOSS: {:.4f}, ACC: {:.2f}, PPL: {:.4f} | "
    #               "ELAPSED TIME: {}\n".
    #               format(self.train_metric_loss.result(),
    #                      self.train_metric_accuracy.result() * 100.0,
    #                      math.exp(self.train_metric_loss.result()),
    #                      self.valid_metric_loss.result(),
    #                      self.valid_metric_accuracy.result() * 100.0,
    #                      math.exp(self.valid_metric_loss.result()),
    #                      format_time(time.time() - train_start_time)))

    def compile_model(self):
        self.compile(optimizer=self.optimizer, loss=self.criterion, metrics=['accuracy'])

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
