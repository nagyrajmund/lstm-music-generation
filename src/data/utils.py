from torch.nn.utils import rnn

def pad_sequences(batch):
    xx = [elem[0][0] for elem in batch]
    yy = [elem[1][0] for elem in batch]
    
    x_lens = [len(x) for x in xx]
    y_lens = [len(y) for y in yy]
    xx_pad = rnn.pad_sequence(xx, batch_first=True, padding_value=0)
    yy_pad = rnn.pad_sequence(yy, batch_first=True, padding_value=0)
    #write this because rnn.py on the bottom window doesn't handle our dataset!

    return xx_pad, yy_pad, x_lens, y_lens

