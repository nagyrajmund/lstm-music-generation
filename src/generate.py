from argparse import ArgumentParser
from pytorch_lightning import Trainer
from network.model import AWD_LSTM
from utils.utils import write_mid_mp3_wav

def build_argument_parser():
    parser = ArgumentParser()
    parser.add_argument('--model_path', type=str, default="../saved_models")
    parser.add_argument('--model_file', type=str, default="model") # TODO
    return parser

if __name__ == "__main__":
    # Parse command-line args
    # TODO read model
    args = build_argument_parser().parse_args()
    generated_ind = model.generate(1, 100, 100)
    ind_to_note = model.dataset.ind_to_note
    notes = [ind_to_note.get(ind) for ind in generated_ind]
    notes = " ".join(notes)
    
    sample_freq = 12
    note_offset = 38
    out = "../dataset/test_dataset/generated"
    fname = "generated.mid"
    chordwise = False
    write_mid_mp3_wav(notes, fname, sample_freq, note_offset, out, chordwise)
    