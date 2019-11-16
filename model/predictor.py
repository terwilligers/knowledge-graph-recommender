import torch

def convert_to_etr(e_to_ix, t_to_ix, r_to_ix, path, length):
    '''
    Converts a path of ids back to the original input format
    -not used for anything right now but could be useful for visualization
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


def predict(model, paths, lengths):
    '''
    -model is our KPRN model
    -outputs score for interaction
    '''

    paths = torch.tensor(paths, dtype=torch.long)
    lengths = torch.tensor(lengths, dtype=torch.long)

    #Sort paths in descending order, required by packing in model
    s_lengths, perm_idx = lengths.sort(0, descending=True)
    s_paths = paths[perm_idx]

    with torch.no_grad():
        tag_scores = model(s_paths, s_lengths)
        pooled_scores = model.weighted_pooling(tag_scores)
        prediction_scores = torch.sigmoid(pooled_scores)
        return prediction_scores
