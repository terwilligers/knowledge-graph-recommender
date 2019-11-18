import pickle
import collections
import random
import constants.consts as consts #import only works when running through recommender.py
from tqdm import tqdm

class PathState:
    def __init__(self, path, length, entities):
        self.length = length
        self.path = path
        self.entities = entities

def get_random_index(nums, max_length):
    index_list = list(range(max_length))
    random.shuffle(index_list)
    return index_list[:nums]

def build_paths(start_user, end_song, song_person, person_song, song_user, user_song):
    '''
    Return a list of paths from a specified user to a specified song of length <= 6
    '''

    paths = []
    sample_nums = 30

    queue = collections.deque()
    start = PathState([[start_user, consts.USER_TYPE, consts.END_REL]], 0, {start_user})
    queue.appendleft(start)
    while len(queue) > 0:
        front = queue.pop()
        entity, type = front.path[-1][0], front.path[-1][1]
        if front.length > 3:
            break
        if entity == end_song:
            paths.append(front.path)
            continue
        if type == consts.USER_TYPE and entity in user_song:
            song_list = user_song[entity]
            index_list = get_random_index(sample_nums, len(song_list))
            for index in index_list:
                song = song_list[index]
                if song not in front.entities and (entity != start_user or song != end_song):
                    new_path = front.path[:]
                    new_path[-1][2] = consts.USER_SONG_REL
                    new_path.append([song, consts.SONG_TYPE, consts.END_REL])
                    new_state = PathState(new_path, front.length + 1, front.entities|{song})
                    queue.appendleft(new_state)

        elif type == consts.SONG_TYPE:
            if entity in song_user:
                user_list = song_user[entity]
                index_list = get_random_index(sample_nums, len(user_list))
                for index in index_list:
                    user = user_list[index]
                    #don't consider user song connection if it exists
                    if user not in front.entities and (entity != end_song or user != start_user):
                        new_path = front.path[:]
                        new_path[-1][2] = consts.SONG_USER_REL
                        new_path.append([user, consts.USER_TYPE, consts.END_REL])
                        new_state = PathState(new_path, front.length + 1, front.entities|{user})
                        queue.appendleft(new_state)
            if entity in song_person:
                person_list = song_person[entity]
                index_list = get_random_index(sample_nums, len(person_list))
                for index in index_list:
                    person = person_list[index]
                    if person not in front.entities:
                        new_path = front.path[:]
                        new_path[-1][2] = consts.SONG_PERSON_REL
                        new_path.append([person, consts.PERSON_TYPE, consts.END_REL])
                        new_state = PathState(new_path, front.length + 1, front.entities|{person})
                        queue.appendleft(new_state)

        elif type == consts.PERSON_TYPE and entity in person_song:
            song_list = person_song[entity]
            index_list = get_random_index(sample_nums, len(song_list))
            for index in index_list:
                song = song_list[index]
                if song not in front.entities:
                    new_path = front.path[:]
                    new_path[-1][2] = consts.PERSON_SONG_REL
                    new_path.append([song, consts.SONG_TYPE, consts.END_REL])
                    new_state = PathState(new_path, front.length + 1, front.entities|{song})
                    queue.appendleft(new_state)

    return paths

def main():
    with open("data/song_data_vocab/song_person_ix.dict", 'rb') as handle:
        song_person = pickle.load(handle)

    with open("data/song_data_vocab/person_song_ix.dict", 'rb') as handle:
        person_song = pickle.load(handle)

    with open("data/song_data_vocab/song_user_ix.dict", 'rb') as handle:
        song_user = pickle.load(handle)

    with open("data/song_data_vocab/user_song_ix.dict", 'rb') as handle:
        user_song = pickle.load(handle)

    print(build_paths(3142, '1302', song_person, person_song, song_user, user_song))


if __name__ == "__main__":
    main()
