# TODO: this should be added to every runnable script to make imports work
# Eventually we can use python
import sys
sys.path.append('../')

from torch import optim, nn, utils, autograd
from torch.utils.data import DataLoader
from pathlib import Path
from data.datasets import ClaraDataset 
from network.awd_lstm import AWD_LSTM

def train_network(config):
    """
    Initialize data set and network.

    Parameters:
        config:  map containing relevant parameters

    Returns:
        model, loss function, optimizer
    """

    # TODO implement
    # train_loader = DataLoader
    # val_loader = DataLoader

    model = AWD_LSTM(\
        ntokens = config['ntokens'], input_size=config['batch_size'], embedding_size=config['embedding_size'], \
            hidden_size=config['hidden_size'], nlayers=config['nlayers'])
    loss_f = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(self.parameters(), lr=config['lr'])
    optimizer = optim.ASGD(model.parameters(), lr=0.01, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)

    # train
    for epoch in range(config['n_epochs']):

        for batch_idx, ((train_data, train_labels), (val_data, val_labels)) in enumerate(zip(train_loader, val_loader)):
            if batch_idx == len(train_dataset) // 2:
                break

        # Forward pass (training set)
        optimizer.zero_grad()
        output = model.forward(train_data)

        loss = loss_f(output, train_labels)

        # Forward pass (validation set)
        output_val = model.forward(val_data)
        loss_val = loss_f(output, val_labels)

        loss.backward()  # Backward pass
        optimizer.step()  # Update the weights

        train_losses.append(loss)
        val_losses.append(loss_val)

        if batch_idx == config['batch_size'] - 1:
            break
        if epoch % 10 == 0:
            print('Epoch: ', epoch + 1, '\t training loss: ', loss,  '\t validation loss: ', loss_val)

    return model, loss_f, optimizer

if __name__ == "__main__":
    # this should be read from a file or cmd line
    dataset_dir = ''
    config = {
        'ntokens': 4,
        'n_epochs': 20,
        'batch_size': 10,
        'embedding_size': 400,
        'hidden_size': 600,
        'nlayers': 4,
        'lr': 0.0001
    }

    train_network(config)