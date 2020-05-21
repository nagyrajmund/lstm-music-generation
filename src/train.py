from argparse import ArgumentParser
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from network.model import AWD_LSTM
from utils.convert import write_mid_mp3_wav
import torch
import pickle

def build_argument_parser():
    parser = ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default="../data/datasets/debug_dataset")
    parser.add_argument('--model_path', type=str, default="../models")
    parser.add_argument('--model_file', type=str, default="model.pth")
    parser = AWD_LSTM.add_model_specific_args(parser) # Add model-specific args
    parser = Trainer.add_argparse_args(parser) # Add ALL training-specific args
    return parser

if __name__ == "__main__":
    # Parse command-line args
    hparams = build_argument_parser().parse_args()
    model = AWD_LSTM(hparams)
    checkpoint_callback = ModelCheckpoint(filepath=hparams.model_path , 
                                          monitor='loss',
                                          mode='min',
                                          save_top_k=1,
                                          period = 10, 
                                          save_weights_only=True)
                                          
    trainer = Trainer.from_argparse_args(hparams, checkpoint_callback=checkpoint_callback)
    trainer.fit(model)

    # Save state dict with parameters
    model_data = {'state_dict': model.state_dict(), 'hparams': hparams}
    torch.save(model_data, hparams.model_path + "/" + hparams.model_file)
