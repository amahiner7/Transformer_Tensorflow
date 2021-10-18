from tensorflow.keras.layers import Layer, LayerNormalization, Dropout
from model.ver1.layers.MultiHeadAttention import MultiHeadAttention
from model.ver1.layers.PositionWiseFeedForward import PositionWiseFeedForward


class EncoderBlock(Layer):
    def __init__(self, d_embed, d_model, d_ff, num_heads, dropout_prob, name="EncoderBlock"):
        super().__init__(name=name)

        self.self_attention_norm = LayerNormalization(epsilon=1e-6)
        self.feed_forward_norm = LayerNormalization(epsilon=1e-6)

        self.self_attention_layer = MultiHeadAttention(
            d_embed=d_embed, d_model=d_model,
            num_heads=num_heads, dropout_prob=dropout_prob)

        self.feed_forward_layer = PositionWiseFeedForward(
            d_embed=d_embed, d_ff=d_ff, dropout_prob=dropout_prob)

        self.dropout = Dropout(rate=dropout_prob)

    def call(self, source, mask):
        """
        :param source : shape (batch_size, seq_len, d_embed)
        :param mask: shape (batch_size, seq_len, seq_len)
        :return output: (batch_size, seq_len, d_embed)
        """
        # Self attention
        self_attention_output, _ = self.self_attention_layer(
            query_embed=source,
            key_embed=source,
            value_embed=source,
            mask=mask)

        # Dropout, Residual connection, Layer Norm
        self_attention_output = self.self_attention_norm(source + self.dropout(self_attention_output))

        # Position wise feed forward
        feed_forward_output = self.feed_forward_layer(self_attention_output)

        # Dropout, Residual connection, Layer Normalization
        output = self.feed_forward_norm(self_attention_output + self.dropout(feed_forward_output))

        return output
