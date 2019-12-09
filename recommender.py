import pickle
import torch
import argparse
import random

import constants.consts as consts
from model import KPRN, train, predict
from data.format import format_paths
from data.path_extraction import build_paths, find_paths_user_to_songs
from eval import hit_at_k, ndcg_at_k

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


def load_train_data(pos_interaction_pairs, song_person, person_song, user_song_all, \
              song_user_train, user_song_train, neg_samples, limit=10):
    '''
    Constructs paths for training data
    '''
    user_to_paths = {}
    user_to_neg_songs_with_paths = {}
    interactions = []
    #current index in negative list for adding negative interactions
    user_to_cur_index = defaultdict(lambda:0)

    for [user,pos_song] in tqdm(pos_interaction_pairs[:limit]):
        if user not in user_to_paths:
            #find paths
            user_to_songs = find_paths_user_to_songs(user, song_person, person_song, song_user_train, user_song_train)
            user_to_paths[user] = user_to_songs

            #select negative paths
            all_pos_songs = set(user_song_all[user])
            songs_with_paths = set(user_to_songs.keys())
            neg_songs_with_paths = list(songs_with_paths.difference(all_pos_songs))
            random.shuffle(neg_songs_with_paths)
            user_to_neg_songs_with_paths[user] = neg_songs_with_paths

        #add paths for positive interaction
        pos_paths = user_to_paths[user][pos_song]
        if len(pos_paths) > 0:
            interactions.append((pos_paths, 1))

        #add negative interactions that have paths (4 for train)
        for i in range(neg_samples):
            index = user_to_cur_index[user]
            #check if not enough neg paths
            if index >= len(user_to_neg_songs_with_paths[user]):
                break
            neg_song = user_to_neg_songs_with_paths[user][index]
            neg_paths = user_to_songs[neg_song]
            interactions.append((neg_paths, 0))
            user_to_cur_index[user] += 1

    return interactions

def load_test_data(pos_interaction_pairs, song_person, person_song, user_song_all, \
              song_user_test, user_song_test, neg_samples, limit=10):
    '''
    Constructs paths for test data, these are stored by user/pos item pair since we evalute for
    each pair
    '''
    #key will be (user, item) pos pair tuple
    #value will contain tuples of (paths, target) for pos pair and sampled neg pairs
    pos_pair_to_interactions = {}

    user_to_paths = {}
    user_to_neg_songs_with_paths = {}
    #current index in negative list for adding negative interactions
    user_to_cur_index = defaultdict(lambda:0)

    pos_paths_not_found = 0
    for [user,pos_song] in tqdm(pos_interaction_pairs[:limit]):
        interactions = []
        if user not in user_to_paths:
            #find paths
            user_to_songs = find_paths_user_to_songs(user, song_person, person_song, song_user_test, user_song_test)
            user_to_paths[user] = user_to_songs

            #select negative paths
            all_pos_songs = set(user_song_all[user])
            songs_with_paths = set(user_to_songs.keys())
            neg_songs_with_paths = list(songs_with_paths.difference(all_pos_songs))
            random.shuffle(neg_songs_with_paths)
            user_to_neg_songs_with_paths[user] = neg_songs_with_paths

        #add paths for positive interaction
        pos_paths = user_to_paths[user][pos_song]
        if len(pos_paths) > 0:
            interactions.append((pos_paths, 1))
        else:
            pos_paths_not_found += 1

        #add negative interactions that have paths (4 for train)
        for i in range(neg_samples):
            index = user_to_cur_index[user]
            if index >= len(user_to_neg_songs_with_paths[user]):
                print("not enough neg paths found")
                break
            neg_song = user_to_neg_songs_with_paths[user][index]
            neg_paths = user_to_songs[neg_song]
            interactions.append((neg_paths, 0))
            user_to_cur_index[user] += 1

        pos_pair_to_interactions[(user, pos_song)] = interactions

    print("number of pos paths attempted to find:", limit)
    print("number of pos paths not found:", pos_paths_not_found)

    return pos_pair_to_interactions


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

            with open('data/song_data_vocab/user_song_train_ix.dict', 'rb') as handle:
                user_song_train = pickle.load(handle)

            with open('data/song_data_vocab/song_user_train_ix.dict', 'rb') as handle:
                song_user_train = pickle.load(handle)

            neg_samples = 4
            #have ran with limit=100000
            training_data = load_train_data(pos_interactions_train, song_person, person_song, \
                                            user_song, song_user_train, user_song_train, neg_samples, limit=10000)

            formatted_train_data = format_paths(training_data, e_to_ix, t_to_ix, r_to_ix, consts.PAD_TOKEN)

            with open('data/path_data/train_interactions.txt', 'wb') as handle:
                pickle.dump(formatted_train_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        model = train(model, formatted_train_data, args.batch_size, args.epochs)

        #Save model to disk
        print("Saving model to disk...")
        torch.save(model.state_dict(), model_path)

    if args.eval:
        #TODO: need to ensure we find paths for all negative interactions, thus we should sample after path finding
        model.load_state_dict(torch.load(model_path))
        model.eval()

        if not args.find_paths:
            with open('data/path_data/pos_pair_to_interactions.dict', 'rb') as handle:
                pos_pair_to_interactions = pickle.load(handle)

        else:
            print("Loading data")
            with open('data/song_data_vocab/user_song_tuple_test_pos_ix.txt', 'rb') as handle:
                pos_interactions_test = pickle.load(handle)

            with open('data/song_data_vocab/user_song_test_ix.dict', 'rb') as handle:
                user_song_test = pickle.load(handle)

            with open('data/song_data_vocab/song_user_test_ix.dict', 'rb') as handle:
                song_user_test = pickle.load(handle)

            print("Finding paths")
            neg_samples = 100
            pos_pair_to_interactions  = load_test_data(pos_interactions_test, song_person, person_song, \
                                                        user_song, song_user_test, user_song_test, neg_samples, limit=100)

            with open('data/path_data/pos_pair_to_interactions.dict', 'wb') as handle:
                pickle.dump(pos_pair_to_interactions, handle, protocol=pickle.HIGHEST_PROTOCOL)

        #predict scores using model for each combination of one pos and 100 neg interactions
        hit_at_k_scores = []
        ndcg_at_k_scores = []
        for pair,interactions in tqdm(pos_pair_to_interactions.items()):
            formatted_test_data = format_paths(interactions, e_to_ix, t_to_ix, r_to_ix, consts.PAD_TOKEN)

            prediction_scores = predict(model, formatted_test_data, args.batch_size)
            target_scores = [x[1] for x in formatted_test_data]

            #merge prediction scores and target scores into tuples, and rank
            merged = list(zip(prediction_scores, target_scores))
            random.shuffle(merged)
            s_merged = sorted(merged, key=lambda x: x[0], reverse=True)

            hit_at_k_scores.append(hit_at_k(s_merged, args.k))
            ndcg_at_k_scores.append(ndcg_at_k(s_merged, args.k))


        print()
        print("Average hit@K scores across users for k={} is {}".format(args.k, mean(hit_at_k_scores)))
        print("Average ndcg@K scores across users for k={} is {}".format(args.k, mean(ndcg_at_k_scores)))



if __name__ == "__main__":
    main()
