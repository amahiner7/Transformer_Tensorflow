import tensorflow as tf
from tensorflow.keras.layers import Layer, Dropout, Embedding
from model.ver1.embedding.PositionalEncoding import PositionalEncoding


class TransformerEmbedding(Layer):
    def __init__(self, vocab_size, d_model, seq_len, dropout_prob):
        super().__init__()

        self.token_embedding = Embedding(input_dim=vocab_size, output_dim=d_model)
        self.position_embedding = PositionalEncoding(d_model, seq_len)
        self.drop_out = Dropout(rate=dropout_prob)
        self.scale = tf.math.sqrt(tf.cast(d_model, tf.float32))

    def forward(self, input):
        token_embedding = self.token_embedding(input)
        position_embedding = self.position_embedding(input)
        output = (token_embedding * self.scale) + position_embedding
        output = self.drop_out(output)

        return output
