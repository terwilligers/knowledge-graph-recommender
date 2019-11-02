import pickle
import collections

#eventually just define these in one place
person_type_idx = 0
user_type_idx = 1
song_type_idx = 2

user_song_rel_idx = 3
song_user_rel_idx = 4
song_person_rel_idx = 5
person_song_rel_idx = 6

class PathState:
    def __init__(self, path, length):
        self.length = length
        self.path = path

def load_dicts():
    '''
    Loads the relation dictionaries
    '''
    with open('song_data/song_person.dict', 'rb') as handle:
        song_person = pickle.load(handle)

    with open('song_data/person_song.dict', 'rb') as handle:
        person_song = pickle.load(handle)

    with open('song_data/song_user.dict', 'rb') as handle:
        song_user = pickle.load(handle)

    with open('song_data/user_song.dict', 'rb') as handle:
        user_song = pickle.load(handle)

    return song_person, person_song, song_user, user_song

def build_paths(user, song):
    '''
    Return a list of paths from a specified user to a specified song of length <= 6

    '''

    paths = []
    song_person, person_song, song_user, user_song = load_dicts()

    queue = collections.deque()
    start = PathState([[user, user_type_idx, None]], 0)
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
                new_path.append([song, song_type_idx, None])
                new_state = PathState(new_path, front.length + 1)
                queue.appendleft(new_state)

        elif type == song_type_idx:
            if entity in song_user:
                for user in song_user[entity]:
                    new_path = front.path[:]
                    new_path[-1][2] = song_user_rel_idx
                    new_path.append([user, user_type_idx, None])
                    new_state = PathState(new_path, front.length + 1)
                    queue.appendleft(new_state)
            if entity in song_person:
                for person in song_person[entity]:
                    new_path = front.path[:]
                    new_path[-1][2] = song_person_rel_idx
                    new_path.append([person, person_type_idx, None])
                    new_state = PathState(new_path, front.length + 1)
                    queue.appendleft(new_state)

        elif type == person_type_idx and entity in person_song:
            for song in person_song[entity]:
                new_path = front.path[:]
                new_path[-1][2] = person_user_rel_idx
                new_path.append([song, song_type_idx, None])
                new_state = PathState(new_path, front.length + 1)
                queue.appendleft(new_state)

        #Todo add other relationship types, maybe put in function

    return paths




def main():
    print(build_paths('+4xaTltGXsNattBO89s5blOroev2V5i1M1kP0ZJpkyA=', 'mZcB03DbF/QTVPEhGqBKqcKRHTUtmJsHbkv9XJGTP1s='))


if __name__ == "__main__":
    main()
