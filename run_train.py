from config.hyper_parameters import *
from data.legacy.load_data import load_data
from model.ver1.Transformer import Transformer

train_data_loader, valid_data_loader = load_data(0.01)

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

model.compile_model()
model.fit(train_data_loader.item,
          validation_data=valid_data_loader.item,
          epochs=NUM_EPOCHS,
          callbacks=model.make_callbacks(),
          verbose=1)

# model.train_on_epoch(train_data_loader=train_data_loader,
#                      valid_data_loader=valid_data_loader,
#                      epochs=NUM_EPOCHS,
#                      log_interval=50)
