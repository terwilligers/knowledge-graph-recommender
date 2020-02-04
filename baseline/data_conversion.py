import pickle
import numpy as np
from random import randint

def convert_for_bpr(pos_list, neg_list):
    '''
    converts pos/neg usersong pair lists into a matrix where every row contains
    101 tuples with the format ((user, song), 1 or 0)
    each row has 1 positive interactions and 100 negative interactions
    '''
    bpr_matrix = []
    total_row = len(pos_list)
    percent = 0
    count = 0
    while (len(neg_list) > 99) and (len(pos_list) != 0):
        row = []
        for i in range(100):
            neg_interaction = neg_list.pop(0)
            row.append((neg_interaction, 0))
        pos_interaction = pos_list.pop(0)
        row.insert(randint(0, 99), (pos_interaction, 1))
        bpr_matrix.append(row)
        count += 1
        if count % (total_row//100) == 0:
            percent += 1
            print(percent, ' percent done')

    # pickle to python2 format
    pickle.dump(bpr_matrix, open("../data/song_test_data/bpr_matrix_test_rs_py2.pkl","wb"), protocol=2)


def main():
    with open("../data/song_test_data/rs_test_pos_interactions.txt", 'rb') as handle:
        test_pos_user_song = pickle.load(handle)
    with open("../data/song_test_data/rs_test_neg_interactions.txt", 'rb') as handle:
        test_neg_user_song = pickle.load(handle)

    convert_for_bpr(test_pos_user_song, test_neg_user_song)
    '''
    pickle:
    with open('bpr_matrix', 'rb') as f:
        x = pickle.load(f)
        print(x.shape)
    '''

if __name__ == "__main__":
    main()
