import pickle
import torch
import argparse
import random
import mmap

import constants.consts as consts
from model import KPRN, train, predict
from data.format import format_paths
from data.path_extraction import find_paths_user_to_songs
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
    parser.add_argument('--model',
                        type=str,
                        default='model.pt',
                        help='name to save or load model from')
    parser.add_argument('--load_checkpoint',
                        default=False,
                        action='store_true',
                        help='whether to load the current model state before training ')
    parser.add_argument('--train_path_file',
                        type=str,
                        default='train_interactions.txt',
                        help='file name to store/load train paths')
    parser.add_argument('--test_path_file',
                        type=str,
                        default='test_interactions.txt',
                        help='file name to store/load test path dictionary')
    parser.add_argument('--train_inter_limit',
                        type=int,
                        default=10000,
                        help='max number of pos train interactions to find paths for')
    parser.add_argument('--test_inter_limit',
                        type=int,
                        default=1000,
                        help='max number of pos test interactions to find paths for')
    parser.add_argument('-e','--epochs',
                        type=int,
                        default=5,
                        help='number of epochs for training model')
    parser.add_argument('-b', '--batch_size',
                        type=int,
                        default=256,
                        help='batch_size')
    parser.add_argument('--test_len_3_sample',
                        type=int,
                        default=50,
                        help='number of connections to sample at each layer when finding length 3 paths')
    parser.add_argument('--test_len_5_sample',
                        type=int,
                        default=11,
                        help='number of connections to sample at each layer when finding length 5 paths')
    parser.add_argument('--not_in_memory',
                        default=False,
                        action='store_true',
                        help='denotes that the path data does not fit in memory')
    parser.add_argument('--lr',
                        type=float,
                        default=.002,
                        help='denotes that the path data does not fit in memory')

    return parser.parse_args()


def load_train_data(song_person, person_song, user_song_all,
              song_user_train, user_song_train, neg_samples, e_to_ix,
              t_to_ix, r_to_ix, train_path_file, limit=10):
    '''
    Constructs paths for training data, writes each formatted interaction to file
    as we find them
    '''
    path_file = open('data/path_data/' + train_path_file, 'w')

    pos_paths_not_found = 0
    total_pos_interactions = 0
    for user,pos_songs in tqdm(list(user_song_train.items())[:limit]):
        total_pos_interactions += len(pos_songs)
        song_to_paths = None
        neg_songs_with_paths = None
        cur_index = 0 #current index in negative list for adding negative interactions
        for pos_song in pos_songs:
            if song_to_paths is None:
                #find paths
                song_to_paths = find_paths_user_to_songs(user, song_person, person_song,
                                                              song_user_train, user_song_train, 3, 50)

                song_to_paths_len5 = find_paths_user_to_songs(user, song_person, person_song,
                                                             song_user_train, user_song_train, 5, 6)

                for song in song_to_paths_len5.keys():
                    song_to_paths[song].extend(song_to_paths_len5[song])

                #select negative paths
                all_pos_songs = set(user_song_all[user])
                songs_with_paths = set(song_to_paths.keys())
                neg_songs_with_paths = list(songs_with_paths.difference(all_pos_songs))
                random.shuffle(neg_songs_with_paths)

            #add paths for positive interaction
            pos_paths = song_to_paths[pos_song]
            if len(pos_paths) > 0:
                interaction = (format_paths(pos_paths, e_to_ix, t_to_ix, r_to_ix), 1)
                path_file.write(repr(interaction) + "\n")
            else:
                pos_paths_not_found += 1

            #add negative interactions that have paths (4 for train)
            for i in range(neg_samples):
                #check if not enough neg paths
                if cur_index >= len(neg_songs_with_paths):
                    print("not enough neg paths")
                    break
                neg_song = neg_songs_with_paths[cur_index]
                neg_paths = song_to_paths[neg_song]
                interaction = (format_paths(neg_paths, e_to_ix, t_to_ix, r_to_ix), 0)
                path_file.write(repr(interaction) + "\n")

                cur_index += 1

    print("number of pos paths attempted to find:", total_pos_interactions)
    print("number of pos paths not found:", pos_paths_not_found)

    path_file.close()
    return


def load_test_data(song_person, person_song, user_song_all,
              song_user_test, user_song_test, neg_samples, e_to_ix,
              t_to_ix, r_to_ix, test_path_file, len_3_sample, len_5_sample, limit=10):
    '''
    Constructs paths for test data, for each combo of a pos paths and 100 neg paths
    we store these in a single line in the file
    '''

    path_file = open('data/path_data/' + test_path_file, 'w')

    pos_paths_not_found = 0
    total_pos_interactions = 0
    for user,pos_songs in tqdm(list(user_song_test.items())[:limit]):
        total_pos_interactions += len(pos_songs)
        song_to_paths = None
        neg_songs_with_paths = None
        cur_index = 0 #current index in negative list for adding negative interactions
        for pos_song in pos_songs:
            interactions = []
            if song_to_paths is None:
                #find paths, **Question: should we be using song_user_test or song_user here???**
                song_to_paths = find_paths_user_to_songs(user, song_person, person_song,
                                                              song_user_test, user_song_test, 3, len_3_sample)

                song_to_paths_len5 = find_paths_user_to_songs(user, song_person, person_song,
                                                             song_user_test, user_song_test, 5, len_5_sample)

                for song in song_to_paths_len5.keys():
                    song_to_paths[song].extend(song_to_paths_len5[song])

                #select negative paths
                all_pos_songs = set(user_song_all[user])
                songs_with_paths = set(song_to_paths.keys())
                neg_songs_with_paths = list(songs_with_paths.difference(all_pos_songs))
                random.shuffle(neg_songs_with_paths)

            #add paths for positive interaction
            pos_paths = song_to_paths[pos_song]
            if len(pos_paths) > 0:
                interactions.append((format_paths(pos_paths, e_to_ix, t_to_ix, r_to_ix), 1))
            else:
                pos_paths_not_found += 1
                #continue here since we don't test interactions with no pos paths,
                continue

            #add negative interactions that have paths (4 for train)
            for i in range(neg_samples):
                #check if not enough neg paths
                if cur_index >= len(neg_songs_with_paths):
                    print("not enough neg paths, only found:", str(i))
                    break
                neg_song = neg_songs_with_paths[cur_index]
                neg_paths = song_to_paths[neg_song]
                interactions.append((format_paths(neg_paths, e_to_ix, t_to_ix, r_to_ix), 0))
                cur_index += 1

            path_file.write(repr(interactions) + "\n")

    print("number of pos paths attempted to find:", total_pos_interactions)
    print("number of pos paths not found:", pos_paths_not_found)

    path_file.close()
    return


def load_string_to_ix_dicts():
    '''
    Loads the dictionaries mapping entity, relation, and type to id
    '''
    data_path = 'data/' + consts.SONG_IX_MAPPING_DIR

    with open(data_path + 'dense_type_to_ix.dict', 'rb') as handle:
        type_to_ix = pickle.load(handle)
    with open(data_path + 'dense_relation_to_ix.dict', 'rb') as handle:
        relation_to_ix = pickle.load(handle)
    with open(data_path + 'dense_entity_to_ix.dict', 'rb') as handle:
        entity_to_ix = pickle.load(handle)

    return type_to_ix, relation_to_ix, entity_to_ix


def load_rel_ix_dicts():
    '''
    Loads the relation dictionaries
    '''
    data_path = 'data/' + consts.SONG_IX_DATA_DIR

    with open(data_path + 'dense_ix_song_person.dict', 'rb') as handle:
        song_person = pickle.load(handle)
    with open(data_path + 'dense_ix_person_song.dict', 'rb') as handle:
        person_song = pickle.load(handle)
    with open(data_path + 'dense_ix_song_user.dict', 'rb') as handle:
        song_user = pickle.load(handle)
    with open(data_path + 'dense_ix_user_song.dict', 'rb') as handle:
        user_song = pickle.load(handle)

    return song_person, person_song, song_user, user_song


def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines


def main():
    '''
    Main function for our graph recommendation project,
    will eventually have command line args for different items
    '''
    print("Main Loaded")
    args = parse_args()
    model_path = "model/" + args.model

    t_to_ix, r_to_ix, e_to_ix = load_string_to_ix_dicts()
    song_person, person_song, song_user, user_song = load_rel_ix_dicts()

    model = KPRN(consts.ENTITY_EMB_DIM, consts.TYPE_EMB_DIM, consts.REL_EMB_DIM, consts.HIDDEN_DIM,
                 len(e_to_ix), len(t_to_ix), len(r_to_ix), consts.TARGET_SIZE)

    data_ix_path = 'data/' + consts.SONG_IX_DATA_DIR

    if args.train:
        print("Training Starting")
        #either load interactions from disk, or run path extraction algorithm
        if args.find_paths:
            print("Finding paths")

            with open(data_ix_path + 'dense_train_ix_user_song.dict', 'rb') as handle:
                user_song_train = pickle.load(handle)
            with open(data_ix_path + 'dense_train_ix_song_user.dict', 'rb') as handle:
                song_user_train = pickle.load(handle)

            load_train_data(song_person, person_song, user_song, song_user_train,
                            user_song_train, consts.NEG_SAMPLES_TRAIN, e_to_ix,
                            t_to_ix, r_to_ix, args.train_path_file, limit=args.train_inter_limit)

        model = train(model, args.train_path_file, args.batch_size, args.epochs,
                     model_path, args.load_checkpoint, args.not_in_memory, args.lr)

    if args.eval:
        print("Evaluation Starting")

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Device is", device)

        checkpoint = torch.load(model_path, map_location=torch.device(device))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        model = model.to(device)

        if args.find_paths:
            print("Finding Paths")

            with open(data_ix_path + 'dense_test_ix_user_song.dict', 'rb') as handle:
                user_song_test = pickle.load(handle)
            with open(data_ix_path + 'dense_test_ix_song_user.dict', 'rb') as handle:
                song_user_test = pickle.load(handle)

            load_test_data(song_person, person_song, user_song, song_user_test, user_song_test,
                            consts.NEG_SAMPLES_TEST, e_to_ix, t_to_ix, r_to_ix, args.test_path_file,
                           args.test_len_3_sample, args.test_len_5_sample, limit=args.test_inter_limit)

        #predict scores using model for each combination of one pos and 100 neg interactions
        hit_at_k_scores = defaultdict(list)
        ndcg_at_k_scores = defaultdict(list)
        max_k = 15

        file_path = 'data/path_data/' + args.test_path_file
        with open(file_path, 'r') as file:
            for line in tqdm(file, total=get_num_lines(file_path)):
                test_interactions = eval(line.rstrip("\n"))
                prediction_scores = predict(model, test_interactions, args.batch_size, device)
                target_scores = [x[1] for x in test_interactions]

                #merge prediction scores and target scores into tuples, and rank
                merged = list(zip(prediction_scores, target_scores))
                random.shuffle(merged)
                s_merged = sorted(merged, key=lambda x: x[0], reverse=True)

                for k in range(1,max_k+1):
                    hit_at_k_scores[k].append(hit_at_k(s_merged, k))
                    ndcg_at_k_scores[k].append(ndcg_at_k(s_merged, k))


        for k in hit_at_k_scores.keys():
            hit_at_ks = hit_at_k_scores[k]
            ndcg_at_ks = ndcg_at_k_scores[k]
            print()
            print(["Average hit@K for k={0} is {1:.2f}".format(k, mean(hit_at_ks))])
            print(["Average ndcg@K for k={0} is {1:.2f}".format(k, mean(ndcg_at_ks))])



if __name__ == "__main__":
    main()
