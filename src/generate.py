from argparse import ArgumentParser
from pytorch_lightning import Trainer
from network.model import AWD_LSTM
from utils.convert import write_mid_mp3_wav

def build_argument_parser():
    parser = ArgumentParser()
    parser.add_argument('--model_path', type=str, default="../saved_models")
    parser.add_argument('--model_file', type=str, default="model.pth")
    parser.add_argument('--output_path', type=str, default="../out")
    parser.add_argument('--output_file', type=str, default="generated.mid")
    parser.add_argument('--random_seed', type=int, default="0")
    parser.add_argument('--input_len', type=int, default="10")
    parser.add_argument('--predic_len', type=int, default="10")
    parser.add_argument('--sample_freq', type=int, default="12")
    parser.add_argument('--note_offset', type=int, default="38")
    parser.add_argument('--chordwise', type=bool, default=False)
    return parser

if __name__ == "__main__":
    # Parse command-line args
    args = build_argument_parser().parse_args()

    # Load model
    parser = ArgumentParser()
    default_params = \
        AWD_LSTM.add_model_specific_args(parser).parse_args() # TODO Placeholder for the constructor parameter, change it
    model = AWD_LSTM(default_params)
    model.load_state_dict(torch.load(args.model_path + "/" + args.model_file))
    model.eval()

    # Generate
    generated_ind = model.generate(args.random_seed, args.input_len, args.predic_len)

    # Convert tokens to notes
    ind_to_note = model.dataset.ind_to_note
    notes = [ind_to_note.get(ind) for ind in generated_ind]
    notes = " ".join(notes)
    
    # Save as midi
    write_mid_mp3_wav(arg.notes, arg.output_file, arg.sample_freq, arg.note_offset, arg.output_path, arg.chordwise)