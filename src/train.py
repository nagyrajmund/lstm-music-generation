from argparse import ArgumentParser
from pytorch_lightning import Trainer
from network.model import AWD_LSTM
import torch

def build_argument_parser():
    parser = ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default="../data/datasets/debug_dataset", help='dataset path')
    parser.add_argument('--model_path', type=str, default="../models", help='model directory')
    parser.add_argument('--model_file', type=str, default="model", help='model file without extension')

    # Parameters for automatic generation
    parser.add_argument('--output_path', type=str, default="../data/generated_outputs/debug_dataset", help='output directory')
    parser.add_argument('--output_name', type=str, default=None, help='file to save output to (without extension)')
    parser.add_argument('--random_seed', type=int, default=None)
    parser.add_argument('--input_len', type=int, default=4)
    parser.add_argument('--predic_len', type=int, default=500)
    parser.add_argument('--sample_freq', type=int, default=4)
    parser.add_argument('--note_offset', type=int, default=38)
    parser.add_argument('--use_chordwise', action='store_true', default=False, help='use bias')
    parser = AWD_LSTM.add_model_specific_args(parser) # Add model-specific args
    parser = Trainer.add_argparse_args(parser) # Add ALL training-specific args
    return parser

if __name__ == "__main__":
    # Parse command-line args
    hparams = build_argument_parser().parse_args()
    model = AWD_LSTM(hparams)
    trainer = Trainer.from_argparse_args(hparams, gradient_clip_val=0.5)
    trainer.fit(model)
