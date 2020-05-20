from argparse import ArgumentParser
from pytorch_lightning import Trainer
from network.model import AWD_LSTM
from utils.convert import write_mid_mp3_wav
import torch

def build_argument_parser():
    parser = ArgumentParser()
    parser.add_argument('--model_path', type=str, default="../models")
    parser.add_argument('--model_file', type=str, default="model.pth")
    parser.add_argument('--output_path', type=str, default="../data/generated_outputs/debug_dataset")
    parser.add_argument('--output_file', type=str, default="generated.mid")
    parser.add_argument('--output_text', type=str, default="generated.txt")
    parser.add_argument('--random_seed', type=int, default="0")
    parser.add_argument('--input_len', type=int, default="100")
    parser.add_argument('--predic_len', type=int, default="100")
    parser.add_argument('--sample_freq', type=int, default="4")
    parser.add_argument('--note_offset', type=int, default="38")
    parser.add_argument('--chordwise', type=bool, default=False)
    return parser

if __name__ == "__main__":
    # Parse command-line args
    args = build_argument_parser()
    args.add_argument('--dataset_path', type=str, default="../data/datasets/debug_dataset")
    
    # Load model
    args = \
        AWD_LSTM.add_model_specific_args(args).parse_args() # TODO Placeholder for the constructor parameter, change it
    model = AWD_LSTM(args)
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(args.model_path + "/" + args.model_file))
    else:
        model.load_state_dict(torch.load(args.model_path + "/" + args.model_file, map_location=torch.device('cpu')))
    model.eval()

    # Generate
    generated_ind = model.generate(args.random_seed, args.input_len, args.predic_len)

    # Convert tokens to notes
    ind_to_note = list(model.dataset.ind_to_note)
    notes = [ind_to_note[ind] for ind in generated_ind]
    notes = " ".join(notes)

    # Save notes as txt
    f = open(args.output_path + "/" + args.output_text, "w")
    f.write(notes) 
    f.close()
    
    # Save as midi
    write_mid_mp3_wav(notes, args.output_file, args.sample_freq, args.note_offset, args.output_path, args.chordwise)