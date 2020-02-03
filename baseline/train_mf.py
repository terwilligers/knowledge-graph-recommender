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
                        help='The type of subnetwork to form from the full KG')
    parser.add_argument("--without_sampling",
                        action="store_true")
    parser.add_argument("--load_pretrained_model",
                        action="store_true")
    parser.add_argument("--do_train",
                        action="store_true")
    args = parser.parse_args()
    return args

def load_data(args):
    # load user song dict
    if args.subnetwork == 'dense':
        with open("../data/song_data_ix/dense_train_ix_user_song_py2.pkl", 'rb') as handle:
            train_user_song = cPickle.load(handle)
        with open("../data/song_data_ix/dense_test_ix_user_song_py2.pkl", 'rb') as handle:
            test_user_song = cPickle.load(handle)
        with open("../data/song_data_ix/dense_ix_song_user_py2.pkl", 'rb') as handle:
            full_song_user = cPickle.load(handle)
    elif args.subnetwork == 'rs':
        with open("../data/song_data_ix/rs_train_ix_user_song_py2.pkl", 'rb') as handle:
            train_user_song = cPickle.load(handle)
        with open("../data/song_data_ix/rs_test_ix_user_song_py2.pkl", 'rb') as handle:
            test_user_song = cPickle.load(handle)
        with open("../data/song_data_ix/rs_ix_song_user_py2.pkl", 'rb') as handle:
            full_song_user = cPickle.load(handle)

    global kprn2matrix_user
    global kprn2matrix_song
    #TODO: get rid of users who don't listen to any songs in subnet training data
    #TODO: get the index correspondence of the kprn data and the matrix indices
    #TODO: use the index conversion in places needed

    # learn user index information
    # note that user in train and test should be exactly the same
    user_ix = list(train_user_song.keys())
    user_ix.sort()
    min_user_ix = user_ix[0]
    max_user_ix = user_ix[-1]
    num_users = max_user_ix - min_user_ix + 1

    # learn song index information
    song_ix = list(full_song_user.keys())
    song_ix.sort()
    min_song_ix = song_ix[0]
    max_song_ix = song_ix[-1]
    num_songs = max_song_ix - min_song_ix + 1

    return train_user_song, test_user_song, full_song_user, full_song_user, \
           num_users, min_user_ix, num_songs, min_song_ix

def prep_train_data(train_user_song, num_users, min_user_ix, num_songs, min_song_ix):
    '''
    prepare training data in csr sparse matrix form
    '''
    # redesignate the user index
    train_user_song_0ix = {}
    for user_ix in train_user_song.keys():
        actual_user_ix = user_ix-min_user_ix
        train_user_song_0ix[actual_user_ix] = []
        for song_ix in train_user_song[user_ix]:
            # note that person song is not relevant in mf, as we do not care about
            # paths, so that songs that only show up in person song don't have an
            # index here
            actual_song_ix = song_ix-min_song_ix
            train_user_song_0ix[actual_user_ix].append(actual_song_ix)

    # convert dictionary to sparse matrix
    mat = sp.dok_matrix((num_users, num_songs), dtype=np.int8)
    print 'number of users: ', num_users
    print 'number of songs: ', num_songs

    count = 0
    for user_id, song_ids in train_user_song_0ix.items():
        count += 1
        for song_id in song_ids:
            mat[user_id, song_id] = 1
    mat = mat.tocsr()
    print 'shape of sparse matrix', mat.shape

    # cut matrix: only select the first 500 users
    #indices=list(xrange(500))
    #mat = mat[:, indices]
    #print 'shape of sparse matrix after cutting', mat.shape
    print 'number of nonzero entries: ', mat.nnz

    return mat

def prep_test_data(test_user_song, train_user_song, full_song_user, min_user_ix, min_song_ix):
    '''
    for each user, for every 1 positive interaction in test data,
    randomly sample 100 negative interactions in tests data

    only evaluate on 10 users here
    '''
    # both test_neg_inter and test_pos_inter are a list of (u, i) pairs
    # where u is actual user index and i is actual song index
    test_neg_inter = [] # don't exist in either train and test
    test_pos_inter = [] # exist in test
    all_eval_users = list(xrange(10))
    # test_data is a list of lists,
    # where each list is a list of 101 pairs ((u,i),tag)
    test_data = []

    print 'find all pos interactions...'
    for actual_user_ix in all_eval_users:
        user_ix = actual_user_ix + min_user_ix
        for song_ix in test_user_song[user_ix]:
            actual_song_ix = song_ix-min_song_ix
            test_pos_inter.append((actual_user_ix, actual_song_ix))

    print 'sample neg interactions...'
    for each_pos in test_pos_inter:
        instance = []
        instance.append((each_pos, 1))
        # append negative pairs
        actual_user_ix = each_pos[0]
        fake_user_ix = actual_user_ix + min_user_ix
        # use fake song ix
        all_songs = set(full_song_user.keys())
        train_pos = set(train_user_song[fake_user_ix])
        test_pos = set(test_user_song[fake_user_ix])
        all_negative_songs = all_songs - train_pos - test_pos

        neg_samples = random.sample(all_negative_songs, 100)
        for neg_song_fake_ix in neg_samples:
            neg_song_actual_ix = neg_song_fake_ix - min_song_ix
            instance.append(((actual_user_ix, neg_song_actual_ix), 0))
        test_data.append(instance)

    return test_data

def evaluate(model, k, test_user_song, train_user_song, full_song_user, \
             min_user_ix, min_song_ix):
    print 'prepare test data...'
    test_data = prep_test_data(test_user_song, train_user_song, full_song_user, \
                               min_user_ix, min_song_ix)
    hit = 0
    ndcg = 0
    total = 0
    for instance in test_data:
        rank_tuples = []
        for i in instance:
            score = model.predict(i[0][0],i[0][1])
            tag = i[1]
            rank_tuples.append((score, tag))
        # sort rank tuples based on descending order of score
        rank_tuples.sort(reverse=True)
        #print('rank_tuples: ', rank_tuples)
        hit = hit + hit_at_k(rank_tuples, k)
        ndcg = ndcg + ndcg_at_k(rank_tuples, k)
        total = total + 1

    print 'Total number of test cases: ', total
    print 'hit at %d: %f' % (k, hit/float(total))
    print 'ndcg at %d: %f' % (k, ndcg/float(total))

def main():
    random.seed(0)
    args = parse_args()

    # load data
    print 'load data...'
    train_user_song, test_user_song, full_song_user, full_song_user, \
    num_users, min_user_ix, num_songs, min_song_ix = load_data(args)

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
        train_data_mat = prep_train_data(train_user_song, num_users, \
                                         min_user_ix, num_songs, min_song_ix)
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
        pickle.dump(model, open(args.output_dir + "/mf_model.pkl","wb"), protocol=2)

    if args.do_train or args.load_pretrained_model:
        print 'evaluating...'
        evaluate(model, args.k, test_user_song, train_user_song, full_song_user, \
                 min_user_ix, min_song_ix)

if __name__=='__main__':
    main()
