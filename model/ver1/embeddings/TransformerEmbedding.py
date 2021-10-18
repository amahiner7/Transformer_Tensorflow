import tensorflow as tf
from tensorflow.keras.layers import Layer, Dropout, Embedding
from model.ver1.embeddings.PositionalEncoding import PositionalEncoding


class TransformerEmbedding(Layer):
    def __init__(self, vocab_size, seq_len, d_embed, dropout_prob, name="TransformerEmbedding"):
        super().__init__(name=name)

        self.token_embedding = Embedding(input_dim=vocab_size, output_dim=d_embed)
        self.position_embedding = PositionalEncoding(max_len=seq_len, d_embed=d_embed)
        self.drop_out = Dropout(rate=dropout_prob)
        self.scale = tf.math.sqrt(tf.cast(d_embed, tf.float32))

    def call(self, input):
        token_embedding = self.token_embedding(input)
        position_embedding = self.position_embedding(input)
        output = (token_embedding * self.scale) + position_embedding
        output = self.drop_out(output)

        return output
