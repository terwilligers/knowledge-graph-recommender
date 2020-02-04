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
    while (len(neg_list) > 99) and (len(pos_list) != 0):
        if len(neg_list) % 15573 <= 100:
            print(percent, "%")
            percent += 1
        row = []
        for i in range(100):
            neg_user = neg_list.pop(0)
            row.append((neg_user, 0))
        pos_user = pos_list.pop(0)
        row.insert(randint(0, 99), (pos_user, 1))
        bpr_matrix.append(row)
    mat = np.array(bpr_matrix)
    with open('bpr_matrix', 'wb') as f:
        pickle.dump(mat, f, protocol=pickle.HIGHEST_PROTOCOL)
        # np.save('bpr_matrix', mat)


def main():
    convert_for_bpr(test_pos_user_song, test_neg_user_song)
    '''
    numpy:
    matrix_python3 = np.load("../baseline/bpr_matrix.npy", allow_pickle=True)
    
    pickle: 
    with open('bpr_matrix', 'rb') as f:
        x = pickle.load(f)
        print(x.shape)
    '''

if __name__ == "__main__":
    main()