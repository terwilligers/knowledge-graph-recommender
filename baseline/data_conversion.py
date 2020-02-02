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

#converts pos/neg usersong pair lists into a matrix where every row contains 101 tuples with the format
# ((song, user) 1 or 0)
# each row has 1 positive interactions and 100 negative interactions
def convert_for_bpr(pos_list, neg_list):
    matrix = []
    percent = 0
    while len(neg_list) > 99:
        if len(neg_list) % 15573 == 0:
            print(percent, "%")
            percent += 1
        row = []
        for i in range(100):
            rand_neg_user = neg_list.pop(randint(0, len(neg_list) - 1))
            row.append(((rand_neg_user[1], rand_neg_user[0]), 0))
        rand_pos_user = pos_list.pop(randint(0, len(pos_list) - 1))
        row.insert(randint(0, 100), ((rand_pos_user[1], rand_pos_user[0]), 1))
        matrix.append(row)
    return matrix
def main():
   print(convert_for_bpr(test_pos_user_song,test_neg_user_song))
if __name__ == "__main__":
    main()