import numpy as np
from scipy.io import mmread, mmwrite
from scipy import sparse
from bpr import *
import cPickle
import numpy as np
import scipy.sparse as sp
import random
import sys
sys.path.append('..')
from eval import hit_at_k, ndcg_at_k

def load_data():
    # load user song dict
    with open("../data/song_data_ix/dense_train_ix_user_song_py2.pkl", 'rb') as handle:
        train_user_song = cPickle.load(handle)
    with open("../data/song_data_ix/dense_test_ix_user_song_py2.pkl", 'rb') as handle:
        test_user_song = cPickle.load(handle)
    with open("../data/song_data_ix/dense_ix_song_user_py2.pkl", 'rb') as handle:
        full_song_user = cPickle.load(handle)

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

def evaluate(model, k, test_user_song, train_user_song, full_song_user, min_user_ix, min_song_ix):
    test_data = prep_test_data(test_user_song, train_user_song, full_song_user, min_user_ix, min_song_ix)
    yes = 0
    count = 0
    for instance in test_data:
        rank_tuples = []
        for i in instance:
            score = model.predict(i[0][0],i[0][1])
            tag = i[1]
            rank_tuples.append((score, tag))
        # sort rank tuples based on descending order of score
        rank_tuples.sort(reverse=True)
        #print('rank_tuples: ', rank_tuples)
        yes = yes + hit_at_k(rank_tuples, k)
        count = count + 1

    print 'Total number of test cases: ', count
    print 'hit at %d: %.3f' % (k, yes/float(count))

def main():
    # load data
    print 'load data...'
    train_user_song, test_user_song, full_song_user, full_song_user, \
    num_users, min_user_ix, num_songs, min_song_ix = load_data()

    do_train = True
    if do_train:
        train_data_mat = prep_train_data(train_user_song, num_users, min_user_ix, num_songs, min_song_ix)
        # TODO: have option to load prev model or start with a new model
        new_model = True
        if new_model:
            args = BPRArgs()
            args.learning_rate = 0.005

            num_factors = 64
            model = BPR(num_factors,args)

            sample_negative_items_empirically = True
            sampler = UniformPairWithoutReplacement(sample_negative_items_empirically)

            num_iters = 2
            max_samples = 1000

            model.train(train_data_mat,sampler,num_iters,max_samples)
        else:
            #TODO: load the model
            #TODO: continue training the loaded model
            pass

    #TODO: have option to save the trained model

    # evaluate
    print 'evaluating...'
    evaluate(model, 15, test_user_song, train_user_song, full_song_user, min_user_ix, min_song_ix)

if __name__=='__main__':
    main()
