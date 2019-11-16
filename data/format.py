import torch

'''
functions used for converting path data into format for KPRN model
'''

def format_train_paths(training_data, e_to_ix, t_to_ix, r_to_ix, padding_token):
    '''
    Pads paths up to max path length, converting each path into tuple
    of (padded_path, path length). Then constructs list of these tuples each in
    a tuple with the interaction tag.
    '''

    max_len = find_max_train_length(training_data)
    formatted_data = []

    for paths, tag in training_data:
        new_paths = []
        for path in paths:
            path_len = len(path)
            pad_path(path, e_to_ix, t_to_ix, r_to_ix, max_len, padding_token)
            new_paths.append((path, path_len))
        formatted_data.append((new_paths, tag))

    return formatted_data

def format_test_paths(paths, e_to_ix, t_to_ix, r_to_ix, padding_token):
    '''
    Returns list of paths padded up to max length, and list of path lengths
    '''

    max_len = max(len(x) for x in paths)
    padded_paths = []
    lengths = []

    for path in paths:
        lengths.append(len(path))
        padded_paths.append(pad_path(path, e_to_ix, t_to_ix, r_to_ix, max_len, padding_token))

    return padded_paths,lengths


def find_max_train_length(data_tuples):
    '''
    Finds max path length in a list of (interaction, target) tuples
    '''
    max_len = 0
    for (paths, _) in data_tuples:
        for path in paths:
            max_len = max(len(path), max_len)
    return max_len



def pad_path(seq, e_to_ix, t_to_ix, r_to_ix, max_len, padding_token):
    '''
    Pads paths up to max path length
    '''
    relation_padding =  r_to_ix[padding_token]
    type_padding = t_to_ix[padding_token]
    entity_padding = e_to_ix[padding_token]

    while len(seq) < max_len:
        seq.append([entity_padding, type_padding, relation_padding])

    return seq
