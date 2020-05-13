import os
import torch
import torchvision
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset, DataLoader


def prepare_datasets(sampling_method, style, note_range, freq, composers):
    
    r = note_range.split('_')[-1].replace("ange", "")
    f = freq.split('_')[-1].replace("req", "")
    
    csv_name = '_'.join([sampling_method, style, r, f]) + '.csv'
    root = "..\..\dataset"
    path = "..\..\dataset\%s\%s\%s\%s\\" %(sampling_method, style, note_range, freq)
    
    for composer in composers:
        sub_path = path + '\\' + composer + '\\'
        
        for file in os.listdir(sub_path):
            with open(sub_path+file, 'r') as infile:
                text = ' '.join(infile.readlines())
            with open(root +'\\'+ csv_name, 'a') as outfile:
                outfile.write(text)
                outfile.write('\n')


class ClaraDataset(Dataset):
    
    def __init__(self, sampling, style, note_range, freq):
        self.sampling = sampling
        self.style = style
        self.note_range = note_range
        self.freq = freq
        
        file = "..\..\dataset" + sampling + '_' + style + '_r' + str(note_range) + '_f' + str(freq) + '.csv'
        enc = OneHotEncoder(handle_unknown='ignore')
        self.x = torch.tensor(enc.fit_transform(pd.read_csv(file, header=None).values).todense())
        self.y = torch.tensor(enc.fit_transform(pd.read_csv(file, header=None).values).todense())

        
    def __getitem__(self, idx):
        return self.x[inx], self.y[idx]
        
    def __len__(self):
        return self.x.shape[0]
        

if __name__ == '__main__':
    
    sampling_method = ['chordwise', 'notewise']
    style = ['chamber', 'jazz']
    note_range = ['note_range38', 'note_range62']
    freq = ['sample_freq4', 'sample_freq12']
    composers = ['bach', 'beethoven', 'brahms', 'chopin', 'debussy', 'dvorak', 'faure', 'grieg', 'handel']
    jazz = ['jazz']

    for samp in sampling_method:
        for stl in style:
            for r in note_range:
                for f in freq:
                    if stl == 'chamber':
                        prepare_datasets(samp, stl, r, f, composers)
                    else:
                        prepare_datasets(samp, stl, r, f, jazz)

                        dataset = dataset('notewise', 'chamber', 62, 4)
    
