from data.legacy.load_data import *
from model.ver1.Transformer import Transformer


model = Transformer(d_input=train_data_loader.encoder_vocab_size,
                    d_output=train_data_loader.decoder_vocab_size,
                    d_embed=EMBED_DIM,
                    d_model=MODEL_DIM,
                    d_ff=FF_DIM,
                    num_heads=NUM_HEADS,
                    num_layers=NUM_LAYERS,
                    dropout_prob=DROPOUT_PROB,
                    seq_len=MAX_SEQ_LEN)

model.summary_model()

model.train_on_epoch(train_data_loader=train_data_loader,
                     valid_data_loader=valid_data_loader,
                     epochs=NUM_EPOCHS,
                     log_interval=100)
