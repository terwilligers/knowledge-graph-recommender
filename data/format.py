import torch

'''
functions used for converting path data into format for KPRN model
'''

def format_paths(training_data, e_to_ix, t_to_ix, r_to_ix, padding_token):
    '''
    Pads paths up to max path length, then converts data to triplets of
    (padded path, tag, path length) replacing path items with ids
    '''

    max_len = find_max_length(training_data)
    formatted_data = []

    for path, tag in training_data:
        formatted_data.append((convert_path_to_ids(path, e_to_ix, t_to_ix, r_to_ix, max_len, padding_token), tag, len(path)))

    return formatted_data


def find_max_length(data_tuples):
    '''
    Finds max path length in a list of (path, target) tuples
    '''
    max_len = 0
    for (path, _) in data_tuples:
        max_len = max(len(path), max_len)
    return max_len


def convert_path_to_ids(seq, e_to_ix, t_to_ix, r_to_ix, max_len, padding_token):
    '''
    Constructs a tensor of item, type, and relation ids from a path
    Pads paths up to max path length
    '''
    relation_padding =  r_to_ix[padding_token]
    type_padding = t_to_ix[padding_token]
    entity_padding = e_to_ix[padding_token]
    id_pairs = []
    for step in seq:
        e,t,r = step[0], step[1], step[2]
        id_pairs.append([e_to_ix[e], t_to_ix[t], r_to_ix[r]])

    while len(id_pairs) < max_len:
        id_pairs.append([entity_padding, type_padding, relation_padding])

    return torch.tensor(id_pairs, dtype=torch.long)
