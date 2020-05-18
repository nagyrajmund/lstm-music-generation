from argparse import ArgumentParser
from pytorch_lightning import Trainer
from network.model import AWD_LSTM
from utils.utils import write_mid_mp3_wav

def build_argument_parser():
    parser = ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default="../dataset/debug_dataset")
    parser.add_argument('--model_path', type=str, default="../saved_models")
    parser.add_argument('--model_file', type=str, default="model") # TODO
    parser = AWD_LSTM.add_model_specific_args(parser) # Add model-specific args
    parser = Trainer.add_argparse_args(parser) # Add ALL training-specific args
    return parser

if __name__ == "__main__":
    # Parse command-line args
    hparams = build_argument_parser().parse_args()
    model = AWD_LSTM(hparams)
    trainer = Trainer.from_argparse_args(hparams)
    trainer.fit(model)

    # TODO Save model