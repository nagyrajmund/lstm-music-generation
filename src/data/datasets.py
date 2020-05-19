import torch
import pandas as pd
import os
from torch import LongTensor
from torch.utils.data import Dataset, DataLoader


class ClaraDataset(Dataset):
    def __init__(self, dataset_path, dicts = None, chunk_size = 1):
        """
        Parameters:
            dataset_path:  path to the dataset as a string
            dicts:  a list containing note_to_ind (dict) and ind_to_note (list)
        """
        if dicts is not None:
            self.note_to_ind = dicts[0]
            self.ind_to_note = dicts[0].keys()
        else:
            built_dicts = self.build_token_dictionaries(self, dataset_path)
            self.note_to_ind = built_dicts[0]
            self.ind_to_note = built_dicts[1]
        
        self.n_tokens = len(self.note_to_ind)
        directory = os.fsencode(dataset_path)
        self.fnames = [os.path.join(directory,fname) for fname in os.listdir(directory) if os.fsdecode(fname).endswith('.txt')]
        self.chunk_size = chunk_size

    def __len__(self):
        return len(self.fnames)
    
    def __getitem__(self, idx):
        # check if slice is given or index
        if isinstance(idx, slice):
            start = idx.start
            stop = idx.stop
            if stop is None: # if end slice not given, go to end of sequence
                stop = len(self.labels)
            if start is None: # if start slice not given, start from beginning of sequence
                start = 0
            idxs = range(start, stop)
        else:
            idxs = [idx]

        # Pack_sequences expects lists of tensors!
        
        sequences = [LongTensor(self.split_to_chunks(self.tokenise_as_numbers(self.fnames[i]))) for i in idxs]        

        # The labels are the same sequences as the inputs, shifted by one to the right
        x = [sequence[:-1] for sequence in sequences] # Exclude last input to ensure that x and y are the same size
        y = [sequence[1:] for sequence in sequences] 
        #TODO: do this in only one comprehension :)
        return x, y

    #TODO: should we store the dataset in memory? then we only have to tokenise once
    #TODO: use better variable names (is token the note (e.g. 'p80') or the number representation)
    #      idea: use note_to_num and not note_to_ind
    
    @staticmethod
    def build_token_dictionaries(self, dataset_path):
        """
        Parameters:
            dataset_path:  path to the dataset as a string
        
        Return:  
            note_to_ind:  dictionary for converting tokens to numbers
            ind_to_note:  list for converting numbers to tokens
        """
        note_to_ind = {
            '<PAD>'   : 0, # 0 will be used as padding
            '<START>' : 1,
            '<END>'   : 2
        }
        #TODO: append and prepend start and eos to the songs

        directory = os.fsencode(dataset_path)
        #TODO: add beginning/end of song tokens
        for file in os.listdir(directory):
            with open(os.path.join(directory, file), 'r', ) as f:
                for token in f.readline().split():
                    if token not in note_to_ind:
                        note_to_ind[token] = len(note_to_ind)

        ind_to_note = note_to_ind.keys()

        return note_to_ind, ind_to_note


    def tokenise_as_numbers(self, fname):
        with open(fname, 'r') as f:
            note_list = f.readline().split() #TODO add support for multi-line txts!
    
            return [self.note_to_ind['<START>']] + [self.note_to_ind[note] for note in note_list] + [self.note_to_ind['<END>']]

    def split_to_chunks(self, data):
        result = []
        for i in range(0, len(data), self.chunk_size):
            result.append(data[i:i + self.chunk_size])
        return result