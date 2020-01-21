import numpy as np
from scipy.io import mmread, mmwrite
from scipy import sparse
from bpr import *
import cPickle
import numpy as np
import scipy.sparse as sp

# load user song dict
with open("../data/song_data_ix/dense_train_ix_user_song_py2.pkl", 'rb') as handle:
    train_user_song = cPickle.load(handle)
with open("../data/song_data_ix/dense_test_ix_user_song_py2.pkl", 'rb') as handle:
    test_user_song = cPickle.load(handle)
with open("../data/song_data_ix/dense_ix_song_user_py2.pkl", 'rb') as handle:
    full_song_user = cPickle.load(handle)

# learn index information
user_ix = list(train_user_song.keys())
user_ix.sort()
min_user_ix = user_ix[0]
max_user_ix = user_ix[-1]
num_users = max_user_ix - min_user_ix + 1
song_ix = list(full_song_user.keys())
song_ix.sort()
min_song_ix = song_ix[0]
max_song_ix = song_ix[-1]
num_songs = max_song_ix - min_song_ix + 1

# redesignate the user index
train_user_song_0ix = {}
for user_ix in train_user_song.keys():
    train_user_song_0ix[user_ix-min_user_ix] = []
    for song_ix in train_user_song[user_ix]:
        train_user_song_0ix[user_ix-min_user_ix].append(song_ix-min_song_ix)

# convert dictionary to sparse matrix
mat = sp.dok_matrix((num_users, num_songs), dtype=np.int8)
print 'number of users: ', num_users
print 'number of songs: ', num_songs

print 'constructing sparse matrix'
count = 0
for user_id, song_ids in train_user_song_0ix.items():
    count += 1
    if count % 300 == 0:
        print(count)
    for song_id in song_ids:
        mat[user_id, song_id] = 1

print 'convert dok to csr'
mat = mat.tocsr()
print 'shape of sparse matrix', mat.shape

data = mat
args = BPRArgs()
args.learning_rate = 0.01

num_factors = 10
model = BPR(num_factors,args)

sample_negative_items_empirically = True
sampler = UniformPairWithoutReplacement(sample_negative_items_empirically)
num_iters = 500
model.train(data,sampler,num_iters)
