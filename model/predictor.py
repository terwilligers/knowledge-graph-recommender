import torch

from torch.utils.data import DataLoader

from model import sort_batch

def convert_to_etr(ix_to_etr, path, length):
    '''
    Converts a path of ids to the actual path using the ix_to_etr dict
    '''
    new_path = []
    for i,step in enumerate(path):
        if i == length:
            break
        new_path.append([ix_to_etr[step[0].item()], ix_to_etr[step[1].item()], ix_to_etr[step[2].item()]])
    return new_path


def scores_from_paths(model, paths, batch_size, ix_to_etr):
    '''
    -model is our KPRN model
    -paths is a list of path lists, each of which consists of tuples of
    (path, tag, path_length), where the path is padded
    -ix_to_etr converts an id to a path item for displaying
    '''
    with torch.no_grad():
        print("paths and scores of form: [loss for tag 0, loss for tag 1]:")
        print()
        test_loader = DataLoader(dataset=paths, batch_size=batch_size, shuffle=False)
        for path_batch, target_batch, lengths in test_loader:
            s_path_batch, s_targets, s_lengths = sort_batch(path_batch, target_batch, lengths)
            tag_scores = model(s_path_batch, s_lengths)

            for i, ix_path in enumerate(s_path_batch):
                print(convert_to_etr(ix_to_etr, ix_path, s_lengths[i]))
                print(tag_scores[i])
