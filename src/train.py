from argparse import ArgumentParser
from pytorch_lightning import Trainer
from network.model import AWD_LSTM
import torch

def build_argument_parser():
    parser = ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default="../data/datasets/debug_dataset", help='dataset path')
    parser.add_argument('--model_path', type=str, default="../models", help='model directory')
    parser.add_argument('--model_file', type=str, default="model", help='model file without extension')
    parser = AWD_LSTM.add_model_specific_args(parser) # Add model-specific args
    parser = Trainer.add_argparse_args(parser) # Add ALL training-specific args
    return parser

if __name__ == "__main__":
    # Parse command-line args
    hparams = build_argument_parser().parse_args()
    model = AWD_LSTM(hparams)
    trainer = Trainer.from_argparse_args(hparams, gradient_clip_val=0.5)
    trainer.fit(model)
