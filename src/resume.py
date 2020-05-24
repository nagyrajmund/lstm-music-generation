from argparse import ArgumentParser
from pytorch_lightning import Trainer
from network.model import AWD_LSTM
import torch

# Script for resuming training.

def build_argument_parser():
    parser = ArgumentParser()
    parser.add_argument('--model_path', type=str, default="../models", help='model directory')
    parser.add_argument('--model_file', type=str, default="model", help='model file without extension')
    parser.add_argument('--new_model_path', type=str, default=None, help='new model directory')
    parser.add_argument('--new_model_file', type=str, default=None, help='new model file without extension')
    parser = AWD_LSTM.add_model_specific_args(parser) # Add model-specific args
    parser = Trainer.add_argparse_args(parser) # Add ALL training-specific args
    return parser

if __name__ == "__main__":
    # Parse command-line args
    args = build_argument_parser().parse_args()
    
    # Set default new model name
    if args.new_model_path is None:
        args.new_model_path = args.model_path
    if args.new_model_file is None:
        args.new_model_file = args.model_file

    # Full path to model
    model_full_path = args.model_path + "/" + args.model_file + ".pth"
    
    # Load model
    if torch.cuda.is_available():
        model_data = torch.load(model_full_path)
        state_dict, hparams = model_data['state_dict'], model_data['hparams']
        model = AWD_LSTM(hparams).cuda()
    else:
        model_data = torch.load(model_full_path, map_location=torch.device('cpu'))
        state_dict, hparams = model_data['state_dict'], model_data['hparams']
        model = AWD_LSTM(hparams)

    model.load_state_dict(state_dict)
    model.eval()

    # Change model name
    model.hparams.model_path = args.new_model_path
    model.hparams.model_file = args.new_model_file

    # Resume training
    trainer = Trainer.from_argparse_args(model.hparams, gradient_clip_val=0.5)
    trainer.fit(model)