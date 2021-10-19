from keras.utils.vis_utils import plot_model
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from config.hyper_parameters import *
from config.model_parameters import *
from config.file_path import *
from data.legacy.load_data import *
from model.ver1.Transformer import Transformer
from tensorflow.keras.optimizers import Adam

model = Transformer(d_input=encoder_vocab_size,
                    d_output=decoder_vocab_size,
                    d_embed=EMBED_DIM,
                    d_model=MODEL_DIM,
                    d_ff=FF_DIM,
                    num_heads=NUM_HEADS,
                    num_layers=NUM_LAYERS,
                    dropout_prob=DROPOUT_PROB,
                    seq_len=MAX_SEQ_LEN)

model.compile(optimizer=Adam(learning_rate=1e-5), loss='sparse_crossentropy', metrics=['accuracy'])
temp_source = tf.random.uniform((BATCH_SIZE, 38), dtype=tf.int64, minval=0, maxval=200)
temp_target = tf.random.uniform((BATCH_SIZE, 36), dtype=tf.int64, minval=0, maxval=200)
temp_out, _ = model(source=temp_source, target=temp_target)
print("temp_out.shape: ", temp_out.shape)

model.summary_model(encoder_input_shape=temp_source.shape[-1],
                    decoder_input_size=temp_target.shape[-1],
                    batch_size=temp_target.shape[0])

model.train_on_epoch(train_dataset=train_dataset, epochs=NUM_EPOCHS)
