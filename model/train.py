import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from model import KPRN

def sort_batch(batch, targets, lengths):
    '''
    sorts a batch of paths by path length, in decreasing order
    '''
    seq_lengths, perm_idx = lengths.sort(0, descending=True)
    seq_tensor = batch[perm_idx]
    target_tensor = targets[perm_idx]
    return seq_tensor, target_tensor, seq_lengths


def train(formatted_data, e_vocab_size, t_vocab_size, r_vocab_size):
    '''
    -trains and outputs a model using the input data
    -formatted_data is a list of path lists, each of which consists of tuples of
    (path, tag, path_length), where the path is padded to ensure same overall length
    '''

    E_EMBEDDING_DIM = 3 #64 in paper
    T_EMBEDDING_DIM = 3 #32 in paper
    R_EMBEDDING_DIM = 3 #32 in paper
    HIDDEN_DIM = 6 #this might be unit number = 256
    TARGET_SIZE = 2

    model = KPRN(E_EMBEDDING_DIM, T_EMBEDDING_DIM, R_EMBEDDING_DIM, HIDDEN_DIM, e_vocab_size,
                 t_vocab_size, r_vocab_size, TARGET_SIZE)
    loss_function = nn.NLLLoss() #negative log likelihood loss
    #loss_function = nn.CrossEntropyLoss() #This seems to work with relu activation but nllloss does not
    #this is because crossEntropyLoss actually automatically adds the softmax layer to normalize results into p-distribution

    # l2 regularization is tuned from {10−5 , 10−4 , 10−3 , 10−2 }, I think this is weight decay
    # Learning rate is found from {0.001, 0.002, 0.01, 0.02} with grid search
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=.001)

    #DataLoader used for batches
    train_loader = DataLoader(dataset=formatted_data, batch_size=3, shuffle=False)

    for epoch in range(300):  # tiny data so 300 epochs
        for path_batch, targets, lengths in train_loader:

            #sort based on path lengths, largest first, so that we can pack paths
            s_path_batch, s_targets, s_lengths = sort_batch(path_batch, targets, lengths)

            #Pytorch accumulates gradients, so we need to clear before each instance
            model.zero_grad()

            #Run the forward pass.
            tag_scores = model(s_path_batch, s_lengths)

            #Compute the loss, gradients, and update the parameters by calling .step()
            loss = loss_function(tag_scores, s_targets)
            loss.backward()
            optimizer.step()

            # print statistics
            if epoch % 30 == 0:
                print("loss is:", loss.item())

    return model
