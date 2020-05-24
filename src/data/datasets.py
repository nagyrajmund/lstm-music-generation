import torch
import pandas as pd
import os
from torch import LongTensor
from torch.utils.data import Dataset, DataLoader
from numpy import cumsum, argmax, argmin, searchsorted

class ClaraDataset(Dataset):
    def __init__(self, dataset_path, chunk_size, batch_size):
        """
        Parameters:
            dataset_path:  path to the dataset as a string
            dicts:  a list containing note_to_num (dict) and num_to_note (list)
        """
        self.chunk_size = chunk_size
        self.batch_size = batch_size

        directory = os.fsencode(dataset_path)
        self.fnames = [os.path.join(directory,fname) for fname in os.listdir(directory) 
                                                     if os.fsdecode(fname).endswith('.txt')]
        
        vocab, inv_vocab, lengths, token_count = self.build_vocabulary()        
        self.note_to_num = vocab
        self.n_tokens    = len(vocab)
        self.num_to_note = inv_vocab
        self.token_count = token_count
        print({k: v for k, v in sorted(token_count.items(), key=lambda item: item[1])})
        self.song_lengths_in_chunks = lengths
        self.cumsum_of_song_lengths = cumsum(lengths) 
        self.num_batches = self.cumsum_of_song_lengths[-1] // batch_size
        self.open_file_ind = None
    def __len__(self):
        # The last element in the cumsum array contains the total number of chunks
        return self.num_batches
    
    def __getitem__(self, batch_idx):
        """ Retrive the chunks for indices in range(start, start+self.batch_size).
        """
        # TODO: the current solution assumes that the chunks are spread out at most among two songs
        # this is a reasonable assumption, but it would be cleaner if we could support any number of files
        first_chunk_global_idx = batch_idx * self.batch_size
        last_chunk_global_idx  = first_chunk_global_idx + self.batch_size - 1
        # Find the file index that corresponds to the requested chunks
        # e.g. if we need chunks for indices between [50:70), then: 
        #   - the first chunk will be in the first file with cumsum_lengths > 50,
        #   - the last chunk will be in the first file with cumsum_lengths > 70
        first_file_ind = searchsorted(self.cumsum_of_song_lengths, first_chunk_global_idx, side='right')
        last_file_ind  = searchsorted(self.cumsum_of_song_lengths, last_chunk_global_idx,  side='right')

        # Suppose the song has cumsum_lengths = 60 and we want chunks 57:59
        #   - our first chunk will be at index 60 - 57 = -3 (3rd song from the end)
        #   - the last chunk will be at index 60-59 = -1
        # (so the chunk indices within the files are in the [-3:-1] closed interval)
        first_chunk_local_idx = first_chunk_global_idx if first_file_ind == 0 else first_chunk_global_idx - self.cumsum_of_song_lengths[first_file_ind-1]
        last_chunk_local_idx  = last_chunk_global_idx  if last_file_ind  == 0 else last_chunk_global_idx  - self.cumsum_of_song_lengths[last_file_ind-1]
        
        # If we need a new file, store all chunks in it (so that we can reuse them in later batches)
        if self.open_file_ind != first_file_ind:
            self.open_file_ind = first_file_ind
            self.open_chunks = self.split_file_to_chunks(self.fnames[first_file_ind]) 
        
        if first_file_ind == last_file_ind:
            # All chunks are in the same file (the one that's currently open)
            chunks = self.open_chunks[first_chunk_local_idx : last_chunk_local_idx+1]
        else:
            # The chunks are spread out between two files
            chunks = self.open_chunks[first_chunk_local_idx:]  # Take all remaining chunks from the first file
            # Open the second file
            self.open_file_ind = last_file_ind
            self.open_chunks = self.split_file_to_chunks(self.fnames[last_file_ind])
            chunks += self.open_chunks[:last_chunk_local_idx+1]
      
        # Chunks is a list of chunks, each chunk being a tensor
        X = torch.stack(chunks)
        # -> X: tensor of shape (batch_size, chunk_size)
        
        # The labels are simply the following chunks. However, it could happen that 
        # the last label is in a new file, we have to account for that.
        last_label_file_ind = searchsorted(self.cumsum_of_song_lengths, last_chunk_global_idx+1,  side='right')
        if last_label_file_ind != last_file_ind:
            # Open and store the new file, then extract the first chunk (which will be the last label)
            self.open_file_ind = last_label_file_ind
            self.open_chunks = self.split_file_to_chunks(self.fnames[last_label_file_ind])
            last_label = self.open_chunks[0]
        else:
            # Otherwise we can just extract the chunk after the last chunk in X
            last_label = self.open_chunks[last_chunk_local_idx+1]

        # Again, chunks is a list of chunks, each chunk being a tensor
        # Last_label is a list containing just the last label (a tensor)
        Y = torch.stack(chunks[1:] + [last_label])
        # (The labels are the same sequences as the inputs, shifted by one to the right)
        return X, Y

    def build_vocabulary(self):
        """
        Parameters:
            dataset_path:  path to the dataset as a string
        
        Return:  
            note_to_num:  dictionary for converting tokens to numbers
            num_to_note:  list for converting numbers to tokens
        """
        note_to_num = {} #TODO: should we have special tokens for START and END?
        token_count = {}
        song_lengths = []

        for file in self.fnames:
            with open(file, 'r') as f:
                song_tokens = f.readline().split()
                song_lengths.append(len(song_tokens))
                # Store previously unseen tokens in the vocabulary
                for token in song_tokens:
                    if token not in note_to_num:
                        note_to_num[token] = len(note_to_num)

                    if token in token_count:
                        token_count[token] += 1
                    else:
                        token_count[token] = 1
        
        num_to_note = list(note_to_num.keys())
        song_lengths_in_chunks = [length // self.chunk_size for length in song_lengths]

        return note_to_num, num_to_note, song_lengths_in_chunks, token_count

    def tokenise_file(self, fname):
        with open(fname, 'r') as f:
            note_list = f.readline().split()
    
            return [self.note_to_num[note] for note in note_list]

    def split_file_to_chunks(self, fname):
        tokens = self.tokenise_file(fname)       
        length = len(tokens)

        chunks = [torch.LongTensor(tokens[i:i+self.chunk_size]) for i in range(0, length - self.chunk_size + 1, self.chunk_size)]
        return chunks
        