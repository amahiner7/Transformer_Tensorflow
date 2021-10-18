import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adam
import time
import math

from model.ver1.layers.Encoder import Encoder
from model.ver1.layers.Decoder import Decoder
from config.file_path import *
from config.hyper_parameters import *
from utils.common import *
from utils.CosineAnnealingWarmUpRestarts import CosineAnnealingWarmUpRestarts
from utils.LearningRateHistory import LearningRateHistory


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
            self.callbacks.append(learning_rate_history)

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

        target_mask = self.make_source_mask(target_sub_mask)

        return tf.maximum(target_sub_mask, target_mask)

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
