import numpy as np
import matplotlib.pyplot as plt
from config.hyper_parameters import *
from config.model_parameters import *
from config.file_path import *
from data.legacy.load_data import *
from model.ver1.Transformer import Transformer

print("done")

model = Transformer(d_input=encoder_vocab_size,
                    d_output=decoder_vocab_size,
                    d_embed=EMBED_DIM,
                    d_model=MODEL_DIM,
                    d_ff=FF_DIM,
                    num_heads=NUM_HEADS,
                    num_layers=NUM_LAYERS,
                    dropout_prob=DROPOUT_PROB,
                    seq_len=MAX_SEQ_LEN)
# # model.build(input_shape=[(BATCH_SIZE, MAX_SEQ_LEN), (BATCH_SIZE, MAX_SEQ_LEN)])
# model.compile(optimizer=Adam(learning_rate=1e-5), loss='sparse_crossentropy', metrics=['accuracy'])
# # model.summary()
