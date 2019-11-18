import pickle
import torch
import argparse

import constants.consts as consts
from model import KPRN, train, predict
from data.format import format_test_paths, format_train_paths
from data.path_extraction import build_paths
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train',
                        default=False,
                        action='store_true',
                        help='whether to train the model')
    parser.add_argument('--eval',
                        default=False,
                        action='store_true',
                        help='whether to evaluate the model')
    parser.add_argument('--model_dir',
                        type=str,
                        default='/model',
                        help='directory to save the model to')
    parser.add_argument('-e','--epochs',
                        type=int,
                        default=1,
                        help='number of epochs for training model')
    parser.add_argument('-b', '--batch_size',
                        type=int,
                        default=5,
                        help='batch_size')

    return parser.parse_args()


def load_sample_data():
    '''
    Constructs a couple fake hardcoded paths for testing
    '''
    with open('data/song_data_vocab/song_person_ix.dict', 'rb') as handle:
        song_person = pickle.load(handle)

    with open('data/song_data_vocab/person_song_ix.dict', 'rb') as handle:
        person_song = pickle.load(handle)

    with open('data/song_data_vocab/user_song_ix.dict', 'rb') as handle:
        user_song = pickle.load(handle)

    user1 = list(user_song.keys())[0]
    user2 = list(user_song.keys())[1]
    song1 = list(song_person.keys())[0]
    song2 = list(song_person.keys())[1]
    song3 = list(song_person.keys())[2]
    person1 = list(person_song.keys())[0]

    #first item in tuple is list of paths, 2nd item is if interaction occured
    training_data = [
        ([[[user1, 1, 2], [song1, 2, 0],
            [person1, 0, 1], [song2, 2, 5]],
          [[user1, 1, 2], [song1, 2, 3],
                [user2, 1, 2], [song2, 2, 5]],
          [[user1, 1, 2], [song2, 2, 5]]], 1),

        ([[[user1, 1, 2], [song1, 2, 0],
            [person1, 0, 1], [song3, 2, 5]],
          [[user1, 1, 2], [song1, 2, 3],
             [user2, 1, 2], [song3, 2, 5]]], 0),

        ([[[user1, 1, 2], [song1, 2, 5]]], 1),
    ]

    return training_data


def load_sample_data2(song_person, person_song, song_user, user_song):
    '''
    Constructs sample data from path algorithm
    '''
    i = 0
    interactions = []
    for user,songs in user_song.items():
        print("user is", user)
        for song in tqdm(songs):
            if i == 10:
                break
            i += 1
            paths = build_paths(user, song, song_person, person_song, song_user, user_song)
            if i % 2 == 0 and len(paths) > 0:
                interactions.append((paths, 0))
            else:
                interactions.append((paths, 1))
        break

    return interactions


def load_string_to_ix_dicts():
    '''
    Loads the dictionaries mapping entity, relation, and type to id
    '''
    with open('data/song_data_vocab/type_to_ix.dict', 'rb') as handle:
        type_to_ix = pickle.load(handle)

    with open('data/song_data_vocab/relation_to_ix.dict', 'rb') as handle:
        relation_to_ix = pickle.load(handle)

    with open('data/song_data_vocab/entity_to_ix.dict', 'rb') as handle:
        entity_to_ix = pickle.load(handle)

    return type_to_ix, relation_to_ix, entity_to_ix


def load_rel_ix_dicts():
    '''
    Loads the relation dictionaries
    '''
    with open("data/song_data_vocab/song_person_ix.dict", 'rb') as handle:
        song_person = pickle.load(handle)

    with open("data/song_data_vocab/person_song_ix.dict", 'rb') as handle:
        person_song = pickle.load(handle)

    with open("data/song_data_vocab/song_user_ix.dict", 'rb') as handle:
        song_user = pickle.load(handle)

    with open("data/song_data_vocab/user_song_ix.dict", 'rb') as handle:
        user_song = pickle.load(handle)

    return song_person, person_song, song_user, user_song


def main():
    '''
    Main function for our graph recommendation project,
    will eventually have command line args for different items
    '''

    args = parse_args()
    model_path = "model/model.pt"

    t_to_ix, r_to_ix, e_to_ix = load_string_to_ix_dicts()
    song_person, person_song, song_user, user_song = load_rel_ix_dicts()

    model = KPRN(consts.ENTITY_EMB_DIM, consts.TYPE_EMB_DIM, consts.REL_EMB_DIM, consts.HIDDEN_DIM, \
                 len(e_to_ix), len(t_to_ix), len(r_to_ix), consts.TARGET_SIZE)

    #TODO: Load dense subgraph here

    if args.train:
        #TODO: Load training interactions based on dense graph

        #sample data is a list of (path_list, target) tuples
        training_data = load_sample_data2(song_person, person_song, song_user, user_song)
        print(training_data)

        formatted_data = format_train_paths(training_data, e_to_ix, t_to_ix, r_to_ix, consts.PAD_TOKEN)
        print(formatted_data)

        model = train(model, formatted_data, args.batch_size, args.epochs)

        #Save model to disk
        torch.save(model.state_dict(), model_path)

    if args.eval:
        model.load_state_dict(torch.load(model_path))
        model.eval()
        #TODO: loop over user item pairs from test set

        #example pair
        user = 3142
        song = 1302
        paths = build_paths(user, song, song_person, person_song, song_user, user_song)
        #currently format_test_paths works on single list of paths but this could change to one list
        #for all test data
        padded_paths, lengths = format_test_paths(paths, e_to_ix, t_to_ix, r_to_ix, consts.PAD_TOKEN)
        scores = predict(model, padded_paths, lengths)
        print(scores)


if __name__ == "__main__":
    main()
