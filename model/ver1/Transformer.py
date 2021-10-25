from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy

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

        self.learning_rate_schedule = CustomSchedule(d_model)
        self.optimizer = Adam(learning_rate=LEARNING_RATE)
        self.training = True

    @tf.function()
    def train_step(self, data):
        # Unpack the data
        source, target = data
        target_input = target[:, :-1]
        target_real = target[:, 1:]

        with tf.GradientTape() as tape:
            # Compute predictions
            predictions, _ = self.call((source, target_input))

            # Updates the metrics tracking the loss
            loss = self.compiled_loss(target_real, predictions, regularization_losses=self.losses)

        # Compute gradients
        trainable_var = self.trainable_variables
        gradients = tape.gradient(loss, trainable_var)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_var))

        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(target_real, predictions)

        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def test_step(self, data):
        # Unpack the data
        source, target = data
        target_input = target[:, :-1]
        target_real = target[:, 1:]

        # Compute predictions
        predictions, _ = self.call((source, target_input))

        # Updates the metrics tracking the loss
        loss = self.compiled_loss(target_real, predictions, regularization_losses=self.losses)

        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(target_real, predictions)

        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def make_callbacks(self):
        callbacks = []

        model_check_point = ModelCheckpoint(filepath=MODEL_FILE_PATH_FIT_FORM,
                                            monitor='val_loss',
                                            save_weights_only=True,
                                            save_best_only=True,
                                            verbose=1)

        tensorboard = TensorBoard(log_dir=TENSORBOARD_LOG_DIR)

        learning_rate_scheduler = CosineAnnealingWarmUpRestarts(initial_learning_rate=LEARNING_RATE,
                                                                first_decay_steps=1,
                                                                alpha=0.0,
                                                                t_mul=2.0,
                                                                m_mul=1.0)
        learning_rate_history = LearningRateHistory(log_dir=TENSORBOARD_LEARNING_RATE_LOG_DIR)

        callbacks.append(model_check_point)
        callbacks.append(tensorboard)
        callbacks.append(learning_rate_scheduler)
        callbacks.append(learning_rate_history)

        return callbacks

    def criterion(self, label, pred):
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

    def call(self, inputs):
        """
        source shape: (batch_size, source_len)
        target shape: (batch_size, target_len)
        """
        source, target = inputs

        # source_mask shape: (batch_size, 1, 1, source_len)
        # target_mask shape: (batch_size, 1, target_len, target_len)
        source_mask = self.make_source_mask(source=source)
        target_mask = self.make_target_mask(target=target)

        # encoder_output shape: (batch_size, source_len, model_dim)
        encoder_output = self.encoder(source=source, mask=source_mask, training=self.training)

        # output shape: (batch_size, target_len, output_dim)
        # attention shape: (batch_size, num_heads, target_len, src_len)
        output, attention = self.decoder(decoder_source=target, decoder_mask=target_mask,
                                         encoder_source=encoder_output, encoder_mask=source_mask,
                                         training=self.training)

        return output, attention

    def train_on_epoch(self, train_data, valid_data, epochs, callbacks, verbose):
        self.training = True
        history = self.fit(train_data,
                           validation_data=valid_data,
                           epochs=epochs,
                           callbacks=callbacks,
                           verbose=verbose)

        return history

    def evaluate_sentence(self, sentence, tokenizer_pt, tokenizer_en, max_seq_len):
        self.training = False
        start_token = [tokenizer_pt.vocab_size]
        end_token = [tokenizer_pt.vocab_size + 1]

        # 입력 문장은 포르투갈어이므로 start 토큰과 end 토큰을 추가합니다.
        sentence = start_token + tokenizer_pt.encode(sentence) + end_token
        encoder_input = tf.expand_dims(sentence, 0)

        # 타겟은 영어이므로 트랜스포머의 첫번째 단어는 영어 start 토큰입니다.
        decoder_input = [tokenizer_en.vocab_size]
        output = tf.expand_dims(decoder_input, 0)

        for i in range(max_seq_len):
            # predictions.shape == (batch_size, seq_len, vocab_size)
            predictions, attention_weights = self.call((encoder_input, output))

            # seq_len 차원에서 마지막 단어를 선택합니다.
            predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

            # predicted_id가 end 토큰과 같으면 결과를 반환합니다.
            if predicted_id == tokenizer_en.vocab_size + 1:
                return tf.squeeze(output, axis=0), attention_weights

            # predicted_id를 디코더의 입력값으로 들어가는 출력값과 연결합니다.
            output = tf.concat([output, predicted_id], axis=-1)

        return tf.squeeze(output, axis=0), attention_weights

    def translate(self, sentence, tokenizer_en):
        result, attention_weights = self.evaluate_sentence(sentence)

        predicted_sentence = tokenizer_en.decode([i for i in result
                                                  if i < tokenizer_en.vocab_size])

        print('Input: {}'.format(sentence))
        print('Predicted translation: {}'.format(predicted_sentence))

    def compile_model(self):
        self.compile(optimizer=self.optimizer, loss=self.criterion, metrics=['accuracy'])

    def build_graph(self, encoder_input_shape, decoder_input_size, batch_size):
        encoder_input = Input(shape=encoder_input_shape, batch_size=batch_size)
        decoder_input = Input(shape=decoder_input_size, batch_size=batch_size)
        return Model(inputs=[encoder_input, decoder_input], outputs=self.call((encoder_input, decoder_input)))

    def summary_model(self):
        temp_source = tf.random.uniform((BATCH_SIZE, 38), dtype=tf.int64, minval=0, maxval=200)
        temp_target = tf.random.uniform((BATCH_SIZE, 36), dtype=tf.int64, minval=0, maxval=200)
        encoder_input_shape = temp_source.shape[-1]
        decoder_input_size = temp_target.shape[-1]
        batch_size = temp_target.shape[0]

        temp_model = self.build_graph(encoder_input_shape=encoder_input_shape,
                                      decoder_input_size=decoder_input_size,
                                      batch_size=batch_size)
        temp_model.summary()
