import pickle
import numpy as np
import scipy.sparse as sp
from random import randint

with open("../data/song_data_ix/rs_train_pos_interactions.txt", 'rb') as handle:
    train_pos_user_song = pickle.load(handle)
with open("../data/song_data_ix/rs_train_neg_interactions.txt", 'rb') as handle:
    train_neg_user_song = pickle.load(handle)
with open("../data/song_data_ix/rs_test_pos_interactions.txt", 'rb') as handle:
    test_pos_user_song = pickle.load(handle)
with open("../data/song_data_ix/rs_test_neg_interactions.txt", 'rb') as handle:
    test_neg_user_song = pickle.load(handle)
with open("../data/song_data_ix/rs_ix_song_user.dict", 'rb')as handle:
    full_song_user = pickle.load(handle)

# converts pos/neg usersong pair lists into a matrix where every row contains 101 tuples with the format
# ((user, song), 1 or 0)
# each row has 1 positive interactions and 100 negative interactions


def convert_for_bpr(pos_list, neg_list):
    bpr_matrix = []
    percent = 0
    while (len(neg_list) > 99) and (len(pos_list) != 0) and (percent < 10):
        if len(neg_list) % 15573 == 0:
            print(percent, "%")
            percent += 1
        row = []
        for i in range(100):
            rand_neg_user = neg_list.pop(randint(0, len(neg_list) - 1))
            row.append((rand_neg_user, 0))
        rand_pos_user = pos_list.pop(randint(0, len(pos_list) - 1))
        row.insert(randint(0, 99), (rand_pos_user, 1))
        bpr_matrix.append(row)
    mat = np.array(bpr_matrix)
    np.save('10_percent_bpr_matrix', mat)


def main():
    convert_for_bpr(test_pos_user_song, test_neg_user_song)
    # example for how to load matrix. Since you are using python 2, I believe you don't need
    # to have the allow_pickle=True in your input.
    # matrix_python3 = np.load("../baseline/bpr_matrix.npy", allow_pickle=True)
    # matrix_python2 = np.load("../baseline/bpr_matrix.npy")


if __name__ == "__main__":

    main()