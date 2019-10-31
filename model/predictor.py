import torch

from torch.utils.data import DataLoader

from model import sort_batch


def convert_to_etr(e_to_ix, t_to_ix, r_to_ix, path, length):
    '''
    Converts a path of ids back to the original input format
    '''
    ix_to_t = {v: k for k, v in t_to_ix.items()}
    ix_to_r = {v: k for k, v in r_to_ix.items()}
    ix_to_e = {v: k for k, v in e_to_ix.items()}
    new_path = []
    for i,step in enumerate(path):
        if i == length:
            break
        new_path.append([ix_to_e[step[0].item()], ix_to_t[step[1].item()], ix_to_r[step[2].item()]])
    return new_path


def scores_from_paths(model, paths, batch_size, e_to_ix, t_to_ix, r_to_ix):
    '''
    -model is our KPRN model
    -paths is a list of path lists, each of which consists of tuples of
    (path, tag, path_length), where the path is padded
    '''
    with torch.no_grad():
        print("paths and scores of form: [loss for tag 0, loss for tag 1]:")
        print()
        test_loader = DataLoader(dataset=paths, batch_size=batch_size, shuffle=False)
        for path_batch, target_batch, lengths in test_loader:
            s_path_batch, s_targets, s_lengths = sort_batch(path_batch, target_batch, lengths)
            tag_scores = model(s_path_batch, s_lengths)

            for i, ix_path in enumerate(s_path_batch):
                print(convert_to_etr(e_to_ix, t_to_ix, r_to_ix, ix_path, s_lengths[i]))
                print(tag_scores[i])
