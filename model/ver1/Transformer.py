import tensorflow as tf
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

        self.callbacks = None
        self.d_model = d_model
        self.optimizer = None

        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        # checkpoint_path = MODEL_FILE_DIR
        # ckpt = tf.train.Checkpoint(transformer=self,
        #                            optimizer=self.optimizer)
        # self.ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    def _check_compile(self):
        if self.optimizer is None:
            self.compile_model()

    @tf.function(input_signature=train_step_signature)
    def _tf_func_train_on_batch(self, source, target):
        target_input = target[:, :-1]
        target_real = target[:, 1:]

        with tf.GradientTape() as tape:
            predictions, _ = self.call(source, target_input)
            loss = self.loss_function(target_real, predictions)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(target_real, predictions)

    def make_callbacks(self, callbacks=None):
        self.callbacks = []

        if callbacks is not None:
            self.callbacks = callbacks
        else:
            model_check_point = ModelCheckpoint(filepath=MODEL_FILE_PATH,
                                                monitor='val_loss',
                                                save_weights_only=True,
                                                save_best_only=True,
                                                verbose=1)

            tensorboard = TensorBoard(log_dir=TENSORBOARD_LOG_DIR)

            learning_rate_scheduler = CosineAnnealingWarmUpRestarts(initial_learning_rate=1e-5,
                                                                    first_decay_steps=1,
                                                                    alpha=0.0,
                                                                    t_mul=2.0,
                                                                    m_mul=1.0)
            learning_rate_history = LearningRateHistory(log_dir=TENSORBOARD_LEARNING_RATE_LOG_DIR)

            self.callbacks.append(model_check_point)
            self.callbacks.append(tensorboard)
            # self.callbacks.append(learning_rate_scheduler)
            # self.callbacks.append(learning_rate_history)

    def loss_function(self, label, pred):
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

    def call(self, source, target):
        """
        source shape: (batch_size, source_len)
        target shape: (batch_size, target_len)
        """
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

    def train_on_batch(self, data_loader, log_interval):
        for batch_index, (source, target) in enumerate(data_loader.item):
            self._tf_func_train_on_batch(source=source, target=target)

            if batch_index % log_interval == 0 and batch_index is not 0:
                print(" Batch: [{}/{}({:.0f}%)] | Train loss: {:.4f}, accuracy: {:.4f}".format(
                    batch_index * len(source),
                    len(data_loader.dataset),
                    100.0 * batch_index / len(data_loader),
                    self.train_loss.result(),
                    self.train_accuracy.result()))

    def train_on_epoch(self, train_data_loader, valid_data_loader, epochs, log_interval=1):
        self._check_compile()

        for epoch in range(epochs):
            print('=============== Training Epochs {} / {} =============== '.format(epoch + 1, epochs))
            train_start_time = time.time()

            self.train_loss.reset_states()
            self.train_accuracy.reset_states()

            self.train_on_batch(data_loader=train_data_loader, log_interval=log_interval)

            # learning_rate = float(tf.keras.backend.get_value(self.optimizer.lr))
            learning_rate = 0

            print("Training elapsed time: {} | Train loss: {:.4f}, PPL: {:.4f} | Val loss: {:.4f}, PPL: {:.4f} | "
                  "Learning rate: {}\n".
                  format(format_time(time.time() - train_start_time),
                         self.train_loss.result(),
                         math.exp(self.train_loss.result()),
                         0.0,  # val_loss,
                         0.0,  # math.exp(val_loss),
                         learning_rate))

    def compile_model(self):
        learning_rate_schedule = CustomSchedule(self.d_model)
        self.optimizer = Adam(learning_rate=learning_rate_schedule, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    def build_graph(self, encoder_input_shape, decoder_input_size, batch_size):
        self._check_compile()
        encoder_input = Input(shape=encoder_input_shape, batch_size=batch_size)
        decoder_input = Input(shape=decoder_input_size, batch_size=batch_size)
        return Model(inputs=[encoder_input, decoder_input], outputs=self.call(encoder_input, decoder_input))

    def summary_model(self, encoder_input_shape, decoder_input_size, batch_size):
        temp_model = self.build_graph(encoder_input_shape=encoder_input_shape,
                                      decoder_input_size=decoder_input_size,
                                      batch_size=batch_size)
        temp_model.summary()
