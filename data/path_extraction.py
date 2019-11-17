import pickle
import collections
import constants.consts as consts #import only works when running through recommender.py

class PathState:
    def __init__(self, path, length):
        self.length = length
        self.path = path

def build_paths(start_user, end_song, song_person, person_song, song_user, user_song):
    '''
    Return a list of paths from a specified user to a specified song of length <= 6
    '''

    paths = []

    queue = collections.deque()
    start = PathState([[start_user, consts.USER_TYPE, consts.END_REL]], 0)
    queue.appendleft(start)
    while len(queue) > 0:
        front = queue.pop()
        entity, type = front.path[-1][0], front.path[-1][1]
        if front.length > 6:
            break
        if entity == end_song:
            paths.append(front.path)
            continue
        if type == consts.USER_TYPE and entity in user_song:
            for song in user_song[entity]:
                #don't consider user song connection if it exists
                #true is placeholder so we don't get empty paths for now
                if True or entity != start_user or song != end_song:
                    new_path = front.path[:]
                    new_path[-1][2] = consts.USER_SONG_REL
                    new_path.append([song, consts.SONG_TYPE, consts.END_REL])
                    new_state = PathState(new_path, front.length + 1)
                    queue.appendleft(new_state)

        elif type == consts.SONG_TYPE:
            if entity in song_user:
                for user in song_user[entity]:
                    #don't consider user song connection if it exists
                    if True or entity != end_song or user != start_user:
                        new_path = front.path[:]
                        new_path[-1][2] = consts.SONG_USER_REL
                        new_path.append([user, consts.USER_TYPE, consts.END_REL])
                        new_state = PathState(new_path, front.length + 1)
                        queue.appendleft(new_state)
            if entity in song_person:
                for person in song_person[entity]:
                    new_path = front.path[:]
                    new_path[-1][2] = consts.SONG_PERSON_REL
                    new_path.append([person, consts.PERSON_TYPE, consts.END_REL])
                    new_state = PathState(new_path, front.length + 1)
                    queue.appendleft(new_state)

        elif type == consts.PERSON_TYPE and entity in person_song:
            for song in person_song[entity]:
                new_path = front.path[:]
                new_path[-1][2] = consts.PERSON_USER_REL
                new_path.append([song, consts.SONG_TYPE, consts.END_REL])
                new_state = PathState(new_path, front.length + 1)
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
