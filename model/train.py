import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from model import KPRN

def format_paths(training_data, e_to_ix, t_to_ix, r_to_ix):
    '''
    Pads paths up to max path length, then
    converts data to triplets of (padded path, tag, path length) also converting to ids
    '''
    #for padding paths want value not in our paths
    PATH_PADDING =  len(t_to_ix) + len(r_to_ix) + len(e_to_ix)
    max_len = find_max_length(training_data)

    formatted_data = []
    for path, tag in training_data:
        formatted_data.append((prepare_path(path, e_to_ix, t_to_ix, r_to_ix, max_len, PATH_PADDING), tag, len(path)))

    return formatted_data


def find_max_length(data):
    '''
    Finds max path length in a list of (path, target) tuples
    '''
    max_len = 0
    for (path, _) in data:
        max_len = max(len(path), max_len)
    return max_len


def prepare_path(seq, e_to_ix, t_to_ix, r_to_ix, max_len, pad_num):
    '''
    Constructs a tensor of item, type, and relation ids from a path
    '''
    id_pairs = []
    for step in seq:
        e,t,r = step[0], step[1], step[2]
        id_pairs.append([len(t_to_ix) + len(r_to_ix) + e_to_ix[e], len(r_to_ix) + t_to_ix[t], r_to_ix[r]])

    while len(id_pairs) < max_len:
        id_pairs.append([pad_num, pad_num, pad_num])

    return torch.tensor(id_pairs, dtype=torch.long)


def sort_batch(batch, targets, lengths):
    '''
    sorts a batch of paths by path length, in decreasing order
    '''
    seq_lengths, perm_idx = lengths.sort(0, descending=True)
    seq_tensor = batch[perm_idx]
    target_tensor = targets[perm_idx]
    return seq_tensor, target_tensor, seq_lengths


def construct_ix_to_etr(e_to_ix, t_to_ix, r_to_ix):
    '''
    Used to convert idx path back to original path
    '''
    mapping = {}
    for e,ix in e_to_ix.items():
        mapping[len(t_to_ix) + len(r_to_ix) + ix] = e
    for t,ix in t_to_ix.items():
        mapping[len(r_to_ix) + + ix] = t
    for r,ix in r_to_ix.items():
        mapping[ix] = r
    return mapping


def train(formatted_data, e_to_ix, t_to_ix, r_to_ix):
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

    vocab_size = len(e_to_ix) + len(t_to_ix) + len(r_to_ix) + 1 #plus 1 for padding
    model = KPRN(E_EMBEDDING_DIM, T_EMBEDDING_DIM, R_EMBEDDING_DIM, HIDDEN_DIM, vocab_size, TARGET_SIZE)
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
