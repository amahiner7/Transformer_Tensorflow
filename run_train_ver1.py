import argparse
from config.hyper_parameters import HyperParameter
from data.legacy.load_data import load_data
from model.ver1.Transformer import Transformer
from config.file_path import make_directories
from utils.common import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train transformer model")
    parser.add_argument("--epochs", help="Training epochs", type=int,
                        required=False, default=HyperParameter.NUM_EPOCHS, metavar="20")

    args = parser.parse_args()
    epochs = args.epochs

    make_directories()

    train_data_loader, valid_data_loader = load_data()

    model = Transformer(d_input=train_data_loader.encoder_vocab_size,
                        d_output=train_data_loader.decoder_vocab_size,
                        d_embed=HyperParameter.EMBED_DIM,
                        d_model=HyperParameter.MODEL_DIM,
                        d_ff=HyperParameter.FF_DIM,
                        num_heads=HyperParameter.NUM_HEADS,
                        num_layers=HyperParameter.NUM_LAYERS,
                        dropout_prob=HyperParameter.DROPOUT_PROB,
                        seq_len=HyperParameter.MAX_SEQ_LEN)
    model.summary_model()

    model.compile_model()
    history = model.train_on_epoch(train_data=train_data_loader.item,
                                   valid_data=valid_data_loader.item,
                                   epochs=epochs,
                                   callbacks=model.make_callbacks(),
                                   verbose=1)

    display_loss(history.history)
