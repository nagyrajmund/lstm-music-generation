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
    parser.add_argument('--output_name', type=str, default=None)
    parser.add_argument('--random_seed', type=int, default="0")
    parser.add_argument('--input_len', type=int, default="100")
    parser.add_argument('--predic_len', type=int, default="100")
    parser.add_argument('--sample_freq', type=int, default="4")
    parser.add_argument('--note_offset', type=int, default="38")
    parser.add_argument('--chordwise', type=bool, default=False)
    return parser

if __name__ == "__main__":
    # Parse command-line args
    args = build_argument_parser().parse_args()
    
    # Set default output file name
    if args.output_name is None:
        args.output_name = args.model_file + "_" + str(args.input_len) + "_" + str(args.predic_len)

    # Full path to model
    model_full_path = args.model_path + "/" + args.model_file
    
    # Load model
    if torch.cuda.is_available():
        model_data = torch.load(model_full_path)
    else:
        model_data = torch.load(model_full_path, map_location=torch.device('cpu'))

    state_dict, hparams = model_data['state_dict'], model_data['hparams']
    model = AWD_LSTM(hparams)
    model.load_state_dict(state_dict)
    model.eval()

    # Generate
    generated_ind = model.generate(args.random_seed, args.input_len, args.predic_len)

    # Convert tokens to notes
    ind_to_note = model.dataset.ind_to_note
    notes = [ind_to_note[ind] for ind in generated_ind] 
    notes = " ".join(notes)

    # Save notes as txt
    f = open(args.output_path + "/" + args.output_name + ".txt", "w")
    f.write(notes) 
    f.close()
    
    # Save as midi
    write_mid_mp3_wav(notes, args.output_name + ".mid", args.sample_freq, args.note_offset, args.output_path, args.chordwise)