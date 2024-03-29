from tensorflow.keras.layers import Layer, Dropout
from model.ver1.embeddings.TransformerEmbedding import TransformerEmbedding
from model.ver1.layers.EncoderBlock import EncoderBlock


class Encoder(Layer):
    def __init__(self,
                 d_input, d_embed, d_model, d_ff, num_layers, num_heads, seq_len,
                 dropout_prob, name="Encoder"):
        super().__init__(name=name)

        self.transformer_embedding = TransformerEmbedding(vocab_size=d_input,
                                                          seq_len=seq_len,
                                                          d_embed=d_embed,
                                                          dropout_prob=dropout_prob)
        self.block_list = [EncoderBlock(d_embed=d_embed,
                                        d_model=d_model,
                                        d_ff=d_ff,
                                        num_heads=num_heads,
                                        dropout_prob=dropout_prob)
                           for _ in range(num_layers)]

        self.dropout = Dropout(dropout_prob)

    def call(self, source, mask, training):
        output = self.dropout(self.transformer_embedding(source), training=training)

        for block in self.block_list:
            output = block(source=output, mask=mask, training=training)

        return output
