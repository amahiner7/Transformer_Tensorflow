from config.hyper_parameters import *
from data.legacy.load_data import load_data
from model.ver2.Transformer import Transformer
from utils.common import *

train_data_loader, valid_data_loader = load_data()

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

history = model.train_on_epoch(train_data=train_data_loader,
                               valid_data=valid_data_loader,
                               epochs=NUM_EPOCHS,
                               log_interval=50)

display_loss(history)
