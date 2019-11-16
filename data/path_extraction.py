#from tqdm import tqdm
import pickle
import collections
import constants.consts as consts #import only works when running through recommender.py

#eventually just define these in one place
person_type_idx = 0
user_type_idx = 1
song_type_idx = 2
user_song_rel_idx = 3
song_user_rel_idx = 4
song_person_rel_idx = 5
person_song_rel_idx = 6
person_user_rel_idx = 7

class PathState:
    def __init__(self, path, length):
        self.length = length
        self.path = path

def build_paths(user, song, song_person, person_song, song_user, user_song):
    '''
    Return a list of paths from a specified user to a specified song of length <= 6

    '''

    paths = []

    #remove user song pair if it exists
    removed = False
    #might consider using sets for this not lists, or another way?
    if user in user_song.keys() and song in user_song[user]:
        user_song[user].remove(song)
        song_user[song].remove(user)
        removed = True

    queue = collections.deque()
    start = PathState([[user, user_type_idx, consts.END_REL]], 0)
    queue.appendleft(start)
    while len(queue) > 0:
        front = queue.pop()
        entity, type = front.path[-1][0], front.path[-1][1]
        if front.length > 6:
            break
        if entity == song:
            paths.append(front.path)
            continue
        if type == user_type_idx and entity in user_song:
            for song in user_song[entity]:
                new_path = front.path[:]
                new_path[-1][2] = user_song_rel_idx
                new_path.append([song, song_type_idx, consts.END_REL])
                new_state = PathState(new_path, front.length + 1)
                queue.appendleft(new_state)

        elif type == song_type_idx:
            if entity in song_user:
                for user in song_user[entity]:
                    new_path = front.path[:]
                    new_path[-1][2] = song_user_rel_idx
                    new_path.append([user, user_type_idx, consts.END_REL])
                    new_state = PathState(new_path, front.length + 1)
                    queue.appendleft(new_state)
            if entity in song_person:
                for person in song_person[entity]:
                    new_path = front.path[:]
                    new_path[-1][2] = song_person_rel_idx
                    new_path.append([person, person_type_idx, consts.END_REL])
                    new_state = PathState(new_path, front.length + 1)
                    queue.appendleft(new_state)

        elif type == person_type_idx and entity in person_song:
            for song in person_song[entity]:
                new_path = front.path[:]
                new_path[-1][2] = person_user_rel_idx
                new_path.append([song, song_type_idx, consts.END_REL])
                new_state = PathState(new_path, front.length + 1)
                queue.appendleft(new_state)

    #if we pass dictionaries as input, want to add back in pair if removed
    if removed:
        user_song[user].append(song)
        song_user[song].append(user)

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
