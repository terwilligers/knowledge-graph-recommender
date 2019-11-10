import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from model import KPRN

class InteractionData(Dataset):
    def __init__(self, formatted_data):
        self.data = formatted_data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def my_collate(batch):
    '''
    Custom dataloader collate function since we have tuples of lists of paths
    '''
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    target = torch.LongTensor(target)
    return [data, target]


def sort_batch(batch, indexes, lengths):
    '''
    sorts a batch of paths by path length, in decreasing order
    '''
    seq_lengths, perm_idx = lengths.sort(0, descending=True)
    seq_tensor = batch[perm_idx]
    indexes_tensor = indexes[perm_idx]
    return seq_tensor, indexes_tensor, seq_lengths


def train(formatted_data, batch_size, epochs, e_vocab_size, t_vocab_size, r_vocab_size):
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
    interaction_data = InteractionData(formatted_data)
    train_loader = DataLoader(dataset=interaction_data, collate_fn = my_collate, batch_size=batch_size, shuffle=False)

    for epoch in range(epochs):
        for interaction_batch, targets in train_loader:
            #construct tensor of all paths in batch, tensor of all lengths, and tensor of interaction id
            paths = []
            lengths = []
            inter_ids = []
            for inter_id, interaction_paths in enumerate(interaction_batch):
                for path, length in interaction_paths:
                    paths.append(path)
                    lengths.append(length)
                inter_ids.extend([inter_id for i in range(len(interaction_paths))])

            inter_ids = torch.tensor(inter_ids, dtype = torch.long)
            paths = torch.tensor(paths, dtype=torch.long)
            lengths = torch.tensor(lengths, dtype=torch.long)


            #sort based on path lengths, largest first, so that we can pack paths
            s_path_batch, s_inter_ids, s_lengths = sort_batch(paths, inter_ids, lengths)

            #Pytorch accumulates gradients, so we need to clear before each instance
            model.zero_grad()

            #Run the forward pass.
            tag_scores = model(s_path_batch, s_lengths)

            #Get averages of scores over interaction id groups (eventually do weighted pooling layer here)
            start = True
            for i in range(len(interaction_batch)):
                #get inds for this interaction
                inter_idxs = (s_inter_ids == i).nonzero().squeeze(1)

                #weighted pooled scores for this interaction
                pooled_score = model.weighted_pooling(tag_scores[inter_idxs])

                if start:
                    #unsqueeze turns it into 2d tensor, so that we can concatenate along existing dim
                    pooled_scores = pooled_score.unsqueeze(0)
                    start = not start
                else:
                    pooled_scores = torch.cat((pooled_scores, pooled_score.unsqueeze(0)), dim=0)

            #Compute the loss, gradients, and update the parameters by calling .step()
            loss = loss_function(pooled_scores, targets)
            loss.backward()
            optimizer.step()

            # print statistics
            if epoch % 10 == 0:
                print(pooled_scores)
                print(targets)
                print("loss is:", loss.item())

    return model
