import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer


class PositionalEncoding(Layer):
    """
    compute sinusoid encoding.
    """

    def __init__(self, max_len, d_embed, name="PositionalEncoding"):
        """
        constructor of sinusoid encoding class

        :param d_embed: dimension of embedding
        :param max_len: max sequence length
        """
        super().__init__(name=name)

        def _get_angle(position, dim, d_embed):
            return position / np.power(10000, 2 * (dim // 2) / d_embed)

        def _get_positional_angle(position, d_embed):
            return [_get_angle(position, dim, d_embed) for dim in range(d_embed)]

        # same size with input matrix (for adding with input matrix)
        self.encoding = np.array(
            [_get_positional_angle(position=position, d_embed=d_embed) for position in range(max_len)],
            dtype=np.float32)

        self.encoding[:, 0::2] = np.sin(self.encoding[:, 0::2])
        self.encoding[:, 1::2] = np.cos(self.encoding[:, 1::2])
        self.encoding = tf.cast(self.encoding, dtype=tf.float32)

    def call(self, input):
        """
        :param input:
        :return:
        """
        position_encoding = self.encoding[:tf.shape(input)[1]]

        return position_encoding

