from tensorflow.keras.layers import Layer, Dense, Dropout
from model.ver1.embeddings.TransformerEmbedding import TransformerEmbedding
from model.ver1.layers.DecoderBlock import DecoderBlock


class Decoder(Layer):
    def __init__(self,
                 d_output, d_embed, d_model, d_ff, num_layers, num_heads, seq_len,
                 dropout_prob, name="Decoder"):
        super().__init__(name=name)

        self.transformer_embedding = TransformerEmbedding(vocab_size=d_output,
                                                          d_model=d_model,
                                                          seq_len=seq_len,
                                                          dropout_prob=dropout_prob)
        self.block_list = [DecoderBlock(d_embed=d_embed,
                                        d_model=d_model,
                                        d_ff=d_ff,
                                        num_heads=num_heads,
                                        dropout_prob=dropout_prob)
                           for _ in range(num_layers)]

        self.generator_fc_layer = Dense(d_output)
        self.dropout = Dropout(dropout_prob)

    def forward(self, decoder_source, decoder_mask, encoder_source, encoder_mask):
        output = self.dropout(self.transformer_embedding(decoder_source))

        for block in self.block_list:
            output, attention_prob = block(decoder_source=output, decoder_mask=decoder_mask,
                                           encoder_source=encoder_source, encoder_mask=encoder_mask)

        output = self.generator_fc_layer(output)

        return output, attention_prob
