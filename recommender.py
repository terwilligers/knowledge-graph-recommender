import pickle
import torch
import argparse

import constants.consts as consts
from model import KPRN, train, predict
from data.format import format_paths
from data.path_extraction import build_paths, find_paths_user_to_songs
from eval import hit_at_k

from tqdm import tqdm
from statistics import mean
from collections import defaultdict

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
    parser.add_argument('--find_paths',
                        default=False,
                        action='store_true',
                        help='whether to find paths (otherwise load from disk)')
    parser.add_argument('--model_dir',
                        type=str,
                        default='/model',
                        help='directory to save the model to')
    parser.add_argument('-e','--epochs',
                        type=int,
                        default=10,
                        help='number of epochs for training model')
    parser.add_argument('-b', '--batch_size',
                        type=int,
                        default=256,
                        help='batch_size')
    parser.add_argument('-k',
                        type=int,
                        default=15,
                        help='When evaluating consider to k')

    return parser.parse_args()

def load_sample_data(song_person, person_song, song_user, user_song):
    '''
    Constructs sample data from path algorithm
    '''
    i = 0
    interactions = []
    for user,songs in user_song.items():
        print("user is", user)
        for song in tqdm(songs):
            if i == 3:
                break
            i += 1
            paths = build_paths(user, song, song_person, person_song, song_user, user_song)
            if i % 2 == 0 and len(paths) > 0:
                interactions.append((paths, 0))
            elif len(paths) > 0:
                interactions.append((paths, 1))
        break

    return interactions


def load_train_data(user_to_paths, interaction_pairs, song_person, person_song, song_user, user_song, inter_value, limit=10):
    '''
    Constructs paths for training data
    '''

    interactions = []
    for [user,song] in tqdm(interaction_pairs[:limit]):
        if user in user_to_paths:
            paths = user_to_paths[user][song]
        else:
            user_to_songs = find_paths_user_to_songs(user, song_person, person_song, song_user, user_song)
            user_to_paths[user] = user_to_songs
            paths = user_to_songs[song]
        if len(paths) > 0:
            interactions.append((paths, inter_value))

    return interactions

def load_test_data(user_to_paths, user_to_interactions, interaction_pairs, song_person, person_song, song_user, user_song, inter_value, limit=10):
    '''
    Constructs paths for test data, these are stored by user since we evalute for a single user
    '''

    for [user,song] in tqdm(interaction_pairs[:limit]):
        if user in user_to_paths:
            paths = user_to_paths[user][song]
        else:
            user_to_songs = find_paths_user_to_songs(user, song_person, person_song, song_user, user_song)
            user_to_paths[user] = user_to_songs
            paths = user_to_songs[song]
        if len(paths) > 0:
            user_to_interactions[user].append((paths, inter_value))


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
    #TODO: convert user_song test to ids
    song_person, person_song, song_user, user_song = load_rel_ix_dicts()

    model = KPRN(consts.ENTITY_EMB_DIM, consts.TYPE_EMB_DIM, consts.REL_EMB_DIM, consts.HIDDEN_DIM, \
                 len(e_to_ix), len(t_to_ix), len(r_to_ix), consts.TARGET_SIZE)


    if args.train:
        #either load interactions from disk, or run path extraction algorithm
        if not args.find_paths:
            with open('data/path_data/train_interactions.txt', 'rb') as handle:
                formatted_train_data = pickle.load(handle)

        else:
            with open('data/song_data_vocab/user_song_tuple_train_pos_ix.txt', 'rb') as handle:
                pos_interactions_train = pickle.load(handle)

            with open('data/song_data_vocab/user_song_tuple_train_neg_ix.txt', 'rb') as handle:
                neg_interactions_train = pickle.load(handle)

            with open('data/song_data_vocab/user_song_train_ix.dict', 'rb') as handle:
                user_song_train = pickle.load(handle)

            with open('data/song_data_vocab/song_user_train_ix.dict', 'rb') as handle:
                song_user_train = pickle.load(handle)

            user_to_paths = defaultdict(list)
            training_data = load_train_data(user_to_paths, pos_interactions_train,  \
                                        song_person, person_song, song_user_train, user_song_train, 1, limit=100000)
            training_data.extend(load_train_data(user_to_paths, neg_interactions_train, song_person, \
                                        person_song, song_user_train, user_song_train, 0, limit=400000))

            formatted_train_data = format_paths(training_data, e_to_ix, t_to_ix, r_to_ix, consts.PAD_TOKEN)

            with open('data/path_data/train_interactions.txt', 'wb') as handle:
                pickle.dump(formatted_train_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        model = train(model, formatted_train_data, args.batch_size, args.epochs)

        #Save model to disk
        print("Saving model to disk...")
        torch.save(model.state_dict(), model_path)

    if args.eval:
        #TODO: this should be split up by user, since
        #for each user we sample 100 neg items, calculate scores for these items (plus pos one)
        # Then hit@k is whether the positive one is in the top k scores
        # hit@ndcg takes into account order a bit more
        #One problem is that if we only consider neg interactions that have paths, then not finding paths makes our model look good
        model.load_state_dict(torch.load(model_path))
        model.eval()

        if not args.find_paths:
            with open('data/path_data/user_to_interactions.dict', 'rb') as handle:
                user_to_interactions = pickle.load(handle)

        else:
            with open('data/song_data_vocab/user_song_tuple_test_pos_ix.txt', 'rb') as handle:
                pos_interactions_test = pickle.load(handle)

            with open('data/song_data_vocab/user_song_tuple_test_neg_ix.txt', 'rb') as handle:
                neg_interactions_test = pickle.load(handle)

            with open('data/song_data_vocab/user_song_test_ix.dict', 'rb') as handle:
                user_song_test = pickle.load(handle)

            with open('data/song_data_vocab/song_user_test_ix.dict', 'rb') as handle:
                song_user_test = pickle.load(handle)

            user_to_paths = defaultdict(list)
            user_to_interactions = defaultdict(list)
            load_test_data(user_to_paths, user_to_interactions, pos_interactions_test,  \
                                        song_person, person_song, song_user_test, user_song_test, 1, limit=5000)
            load_test_data(user_to_paths, user_to_interactions, neg_interactions_test, song_person, \
                                        person_song, song_user_test, user_song_test, 0, limit=500000)

            with open('data/path_data/user_to_interactions.dict', 'wb') as handle:
                pickle.dump(user_to_interactions, handle, protocol=pickle.HIGHEST_PROTOCOL)

        #predict scores using model
        hit_at_k_scores = []
        for user, interaction_data in user_to_interactions.items():
            formatted_test_data = format_paths(interaction_data, e_to_ix, t_to_ix, r_to_ix, consts.PAD_TOKEN)

            prediction_scores = predict(model, formatted_test_data, args.batch_size)
            target_scores = [x[1] for x in formatted_test_data]

            #merge prediction scores and target scores into tuples, and rank
            merged = list(zip(prediction_scores, target_scores))
            s_merged = sorted(merged, key=lambda x: abs(x[0]))
            print(s_merged[:args.k])

            hit_at_k_scores.append(hit_at_k(s_merged, args.k))

        print()
        print("Average hit@K scores across users for k={} is {}".format(args.k, mean(hit_at_k_scores)))



if __name__ == "__main__":
    main()
