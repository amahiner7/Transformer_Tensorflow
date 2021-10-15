import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Dropout


class MultiHeadAttention(Layer):
    def __init__(self, d_embed, d_model, num_heads, dropout_prob, name="MultiHeadAttention"):
        super().__init__(name=name)
        self.d_model = d_model  # Model dimension = d_key * num_heads
        self.d_embed = d_embed  # Embedding dimension
        self.num_heads = num_heads  # Num of heads

        assert d_model % self.num_heads == 0
        self.d_key = d_model // num_heads  # Key(=Query=Value) dimension

        self.query_layer = Dense(units=d_model)  # Query fully connected layer
        self.key_layer = Dense(units=d_model)  # Key fully connected layer
        self.value_layer = Dense(units=d_model)  # Value fully connected layer

        self.output_layer = Dense(units=d_embed)

        self.dropout = Dropout(rate=dropout_prob)
        self.d_k_scale = tf.math.sqrt(tf.cast(self.d_key, tf.float32))

    def _scale_dot_product_attention(self, query_embed, key_embed, value_embed, mask=None):
        """
        :param query_embed: shape (num_batch, seq_len, d_embed)
        :param key_embed: shape (num_batch, seq_len, d_embed)
        :param value_embed: shape (num_batch, seq_len, d_embed)
        :param mask: shape (num_batch, seq_len, seq_len)
        :return query_attention: shape (num_batch, num_heads, seq_len, d_model)
                attention_prob: (num_batch, num_heads, seq_len, seq_len)
        """

        batch_size = tf.shape(query_embed)[0]
        query = self.query_layer(query_embed)  # shape: (num_batch, seq_len, d_model)
        key = self.key_layer(key_embed)  # shape: (num_batch, seq_len, d_model)
        value = self.value_layer(value_embed)  # shape: (num_batch, seq_len, d_model)

        # query shape: (num_batch, seq_len, num_heads, d_key)
        query = tf.reshape(query, shape=(batch_size, -1, self.num_heads, self.d_key))
        # key shape: (num_batch, seq_len, num_heads, d_key)
        key = tf.reshape(key, shape=(batch_size, -1, self.num_heads, self.d_key))
        # value shape: (num_batch, seq_len, num_heads, d_key)
        value = tf.reshape(value, shape=(batch_size, -1, self.num_heads, self.d_key))

        # query shape: (num_batch, num_heads, seq_len, d_key)
        query = tf.transpose(query, perm=[0, 2, 1, 3])
        # key shape: (num_batch, num_heads, seq_len, d_key)
        key = tf.transpose(key, perm=[0, 2, 1, 3])
        # value shape: (num_batch, num_heads, seq_len, d_key)
        value = tf.transpose(value, perm=[0, 2, 1, 3])

        # attention_score shape: (num_batch, num_heads, seq_len, seq_len)
        attention_score = tf.matmul(query, key, transpose_b=True)
        attention_score = attention_score / self.d_k_scale  # scaling

        if mask is not None:
            attention_score = (mask * -1e10)

        # attention_prob shape: (num_batch, num_heads, seq_len, seq_len), Softmax probability
        attention_prob = tf.nn.softmax(attention_score, axis=-1)
        attention_prob = self.dropout(attention_prob)  # dropout

        # query_attention shape: (num_batch, num_heads, seq_len, d_key)
        query_attention = tf.matmul(attention_prob, value)
        # query_attention shape: (num_batch, seq_len, num_heads, d_key)
        query_attention = tf.transpose(query_attention, perm=[0, 2, 1, 3])
        # query_attention shape: (num_batch, seq_len, d_model)
        query_attention = tf.reshape(query_attention, shape=(batch_size, -1, self.d_model))

        return query_attention, attention_prob

    def call(self, query_embed, key_embed, value_embed, mask=None):
        # query_attention shape: (num_batch, seq_len, d_model)
        query_attention, attention_prob = self._scale_dot_product_attention(query_embed=query_embed,
                                                                            key_embed=key_embed,
                                                                            value_embed=value_embed,
                                                                            mask=mask)

        # output shape: (num_batch, seq_len, d_embed)
        output = self.output_layer(query_attention)

        return output, attention_prob
