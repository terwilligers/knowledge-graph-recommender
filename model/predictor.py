import torch

from torch.utils.data import DataLoader

from model import sort_batch


def convert_to_etr(e_to_ix, t_to_ix, r_to_ix, path, length):
    '''
    Converts a path of ids back to the original input format
    TODO: eventually just use saved inverse dictionaries for this
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


def predict(model, user, entity):
    '''
    -model is our KPRN model
    -outputs score for interaction (will call model with torch.no_grad())
    '''
    pass
