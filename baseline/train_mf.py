import numpy as np
from scipy.io import mmread, mmwrite
from scipy import sparse
from bpr import *
import cPickle
import numpy as np
import scipy.sparse as sp
import random
import sys
import argparse
import pickle
sys.path.append('..')
from eval import hit_at_k, ndcg_at_k
sys.path.append('../data')
from data_preparation import create_directory

# maps the indices in the kprn data to the matrix indices here
kprn2matrix_user = {}
kprn2matrix_song = {}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate",
                        default=0.005,
                        help="learning rate",
                        type=float)
    parser.add_argument("--num_factors",
                        default=64,
                        help="rank of the matrix decomposition",
                        type=int)
    parser.add_argument("--num_iters",
                        default=10,
                        help="number of iterations",
                        type=int)
    parser.add_argument("--max_samples",
                        default=1000,
                        help="max number of training samples in each iteration",
                        type=int)
    parser.add_argument("--k",
                        default=15,
                        help="k for hit@k and ndcg@k",
                        type=int)
    parser.add_argument("--output_dir",
                        default=".",
                        help="save the trained model here",
                        type=str)
    parser.add_argument("--pretrained_dir",
                        default=".",
                        help="load the pretrained model here",
                        type=str)
    parser.add_argument('--subnetwork',
                        default='dense',
                        choices=['dense', 'rs'],
                        help='the type of subnetwork to form from the full KG')
    parser.add_argument("--without_sampling",
                        action="store_true")
    parser.add_argument("--load_pretrained_model",
                        action="store_true")
    parser.add_argument("--do_train",
                        action="store_true")
    parser.add_argument("--do_eval",
                        action="store_true")
    parser.add_argument("--eval_data",
                        default='kprn_test',
                        choices=['kprn_test', 'kprn_test_subset_1000','10users'],
                        help='Evaluation data')
    args = parser.parse_args()
    return args


def load_data(args):
    # load user song dict and song user dict
    if args.subnetwork == 'dense':
        with open("../data/song_data_ix/dense_train_ix_user_song_py2.pkl", 'rb') as handle:
            train_user_song = cPickle.load(handle)
        with open("../data/song_data_ix/dense_test_ix_user_song_py2.pkl", 'rb') as handle:
            test_user_song = cPickle.load(handle)
        with open("../data/song_data_ix/dense_ix_song_user_py2.pkl", 'rb') as handle:
            full_song_user = cPickle.load(handle)
        with open("../data/song_data_ix/dense_ix_song_person_py2.pkl", 'rb') as handle:
            full_song_person = cPickle.load(handle)
    elif args.subnetwork == 'rs':
        print 'rs subnetwork'
        with open("../data/song_data_ix/rs_train_ix_user_song_py2.pkl", 'rb') as handle:
            train_user_song = cPickle.load(handle)
        with open("../data/song_data_ix/rs_test_ix_user_song_py2.pkl", 'rb') as handle:
            test_user_song = cPickle.load(handle)
        with open("../data/song_data_ix/rs_ix_song_user_py2.pkl", 'rb') as handle:
            full_song_user = cPickle.load(handle)
        with open("../data/song_data_ix/rs_ix_song_person_py2.pkl", 'rb') as handle:
            full_song_person = cPickle.load(handle)

    # get rid of users who don't listen to any songs in the subnetwork
    # get rid of them in both train and test user song dictionaries
    for user in train_user_song.keys():
        if train_user_song[user] == None or len(train_user_song[user]) == 0:
            train_user_song.pop(user)
            if user in test_user_song:
                test_user_song.pop(user)

    #get the index correspondence of the kprn data and the matrix indices
    user_ix = list(train_user_song.keys())
    user_ix.sort() # ascending order
    songs_from_user = set(full_song_user.keys())
    songs_from_person = set(full_song_person.keys())
    song_ix = list(songs_from_user.union(songs_from_person))
    song_ix.sort() # ascending order

    return train_user_song, test_user_song, full_song_user, full_song_person, user_ix, song_ix


def prep_train_data(train_user_song, user_ix, song_ix):
    '''
    prepare training data in csr sparse matrix form
    '''
    num_users = len(user_ix)
    num_songs = len(song_ix)

    # convert dictionary to sparse matrix
    mat = sp.dok_matrix((len(user_ix), len(song_ix)), dtype=np.int8)
    print 'number of users: ', num_users
    print 'number of songs: ', num_songs

    for kprn_user_id, kprn_song_ids in train_user_song.items():
        mf_user_id = user_ix.index(kprn_user_id)
        for kprn_song_id in kprn_song_ids:
            mf_song_id = song_ix.index(kprn_song_id)
            mat[mf_user_id, mf_song_id] = 1
    mat = mat.tocsr()
    print 'shape of sparse matrix', mat.shape

    print 'number of nonzero entries: ', mat.nnz

    return mat


def prep_test_data(test_user_song, train_user_song, full_song_user, full_song_person, user_ix, song_ix):
    '''
    for each user, for every 1 positive interaction in test data,
    randomly sample 100 negative interactions in tests data

    only evaluate on 10 users here

    converts kprn indices to mf indices
    '''
    # both test_neg_inter and test_pos_inter are a list of (u, i) pairs
    # where u is actual user index and i is actual song index
    test_neg_inter = [] # don't exist in either train and test
    test_pos_inter = [] # exist in test
    eval_user_ix_mf = list(xrange(10))
    # test_data is a list of lists,
    # where each list is a list of 101 pairs ((u,i),tag)
    test_data = []

    print 'find all pos interactions...'
    for user_ix_mf in eval_user_ix_mf:
        user_ix_kprn = user_ix[user_ix_mf]
        for song_ix_kprn in test_user_song[user_ix_kprn]:
            song_ix_mf = song_ix.index(song_ix_kprn)
            test_pos_inter.append((user_ix_mf, song_ix_mf))

    print 'sample neg interactions...'
    for each_pos in test_pos_inter:
        instance = []
        instance.append((each_pos, 1))
        # append negative pairs
        user_ix_mf = each_pos[0]
        user_ix_kprn = user_ix[user_ix_mf]
        # use user_ix_kprn to find all negative test songs for that user
        all_songs = set(full_song_user.keys()).union(set(full_song_person.keys()))
        train_pos = set(train_user_song[user_ix_kprn])
        test_pos = set(test_user_song[user_ix_kprn])
        all_negative_songs = all_songs - train_pos - test_pos

        neg_samples = random.sample(all_negative_songs, 100)
        for song_ix_kprn in neg_samples:
            song_ix_mf = song_ix.index(song_ix_kprn)
            instance.append(((user_ix_mf, song_ix_mf), 0))
        test_data.append(instance)

    return test_data


def evaluate(args, model, user_ix, song_ix, test_data):
    hit = 0
    ndcg = 0
    total = 0
    user_set_mf = set()
    user_set_kprn = set()
    for instance in test_data:
        rank_tuples = []
        for i in instance:
            tag = i[1]
            if args.eval_data in ['kprn_test_subset_1000', 'kprn_test']:
                #convert kprn indices to mf indices (user and song)
                user_ix_kprn = i[0][0]
                song_ix_kprn = i[0][1]
                user_ix_mf = user_ix.index(user_ix_kprn)
                user_set_mf.add(user_ix_mf)
                user_set_kprn.add(user_ix_kprn)
                song_ix_mf = song_ix.index(song_ix_kprn)
            else:
                user_ix_mf = i[0][0]
                song_ix_mf = i[0][1]
            score = model.predict(user_ix_mf, song_ix_mf)
            rank_tuples.append((score, tag))
        # sort rank tuples based on descending order of score
        rank_tuples.sort(reverse=True)
        hit = hit + hit_at_k(rank_tuples, args.k)
        ndcg = ndcg + ndcg_at_k(rank_tuples, args.k)
        total = total + 1

    print 'Total number of test cases: ', total
    print 'hit at %d: %f' % (args.k, hit/float(total))
    print 'ndcg at %d: %f' % (args.k, ndcg/float(total))


def load_test_data(args):
    test_data = None
    if args.subnetwork == 'dense' and args.eval_data in ['kprn_test_subset_1000', 'kprn_test']:
        with open("../data/song_test_data/bpr_matrix_test_dense_py2.pkl", 'rb') as handle:
            test_data = cPickle.load(handle)
    elif args.subnetwork == 'rs' and args.eval_data in ['kprn_test_subset_1000', 'kprn_test']:
        with open("../data/song_test_data/bpr_matrix_test_rs_py2.pkl", 'rb') as handle:
            test_data = cPickle.load(handle)

    if args.eval_data == 'kprn_test_subset_1000':
        return random.sample(test_data, 1000)
    return test_data


def main():
    random.seed(0)
    args = parse_args()

    # load data
    print 'load data...'
    train_user_song, test_user_song, full_song_user, full_song_person, \
    user_ix, song_ix = load_data(args)

    model = None
    if args.load_pretrained_model:
        with open(args.pretrained_dir + "/mf_model.pkl", 'rb') as handle:
            model = cPickle.load(handle)
    else:
        # initialize a new model
        bpra_args = BPRArgs()
        bpra_args.learning_rate = args.learning_rate
        model = BPR(args.num_factors, bpra_args)

    if args.do_train:
        print 'prepare training data...'
        train_data_mat = prep_train_data(train_user_song, user_ix, song_ix)
        sample_negative_items_empirically = True
        sampler = UniformPairWithoutReplacement(sample_negative_items_empirically)
        max_samples = None if args.without_sampling else args.max_samples
        print 'training...'
        print 'max_samples: ', max_samples
        train_new_model = not args.load_pretrained_model
        model.train(train_data_mat, sampler, args.num_iters, train_new_model, max_samples)

        # save the trained model
        # note that output_dir should contain information about
        # number of iterations, max sample size, num_factors, and learning rate
        create_directory(args.output_dir)
        pickle.dump(model, open(args.output_dir + "/mf_model.pkl","wb"), protocol=2)

    if args.do_train or args.load_pretrained_model and args.do_eval:
        print 'prepare test data...'
        # note: the user and song indices have not been converted to the mf indices
        # the conversion will be done in the evaluate function
        if args.eval_data in ['kprn_test_subset_1000', 'kprn_test']:
            test_data = load_test_data(args)
        elif args.eval_data == '10users':
            test_data = prep_test_data(test_user_song, train_user_song, \
                                       full_song_user, full_song_person, \
                                       user_ix, song_ix)
        print 'evaluating...'
        evaluate(args, model, user_ix, song_ix, test_data)

if __name__=='__main__':
    main()
