from tensorflow.keras.layers import Layer, Dense, ReLU, Dropout


class PositionWiseFeedForward(Layer):
    def __init__(self, d_embed, d_ff, dropout_prob, name="PositionWiseFeedForward"):
        super().__init__(name=name)

        self.first_fc_layer = Dense(units=d_ff)
        self.second_fc_layer = Dense(units=d_embed)
        self.relu = ReLU()
        self.dropout = Dropout(rate=dropout_prob)

    def call(self, input, training):
        """
        :param input: shape (batch_size, seq_len, d_embed)
        :return output: shape (batch_size, seq_len, d_embed)
        """

        output = self.first_fc_layer(input)  # output shape : (batch_size, seq_len, d_ff)
        output = self.relu(output)
        output = self.dropout(output, training=training)

        # output shape : (batch_size, seq_len, d_embed)
        output = self.second_fc_layer(output)

        return output
