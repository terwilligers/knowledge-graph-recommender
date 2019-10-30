from model import KPRN
from model import train
from model import format_paths
from model import construct_ix_to_etr
from model import scores_from_paths

def main():
    '''
    Main function for our graph recommendation project,
    will eventually have command line args for different tasks
    '''
    #For now some hardcoded paths
    training_data = [
        ([['Sam', 'u', 'rate'], ['Song1', 's', 'category'], ['Pop', 't', '_belong'], ['Song2', 's', 'UNK']], 1),
        ([['Sam', 'u', 'rate'], ['Song2', 's', 'UNK']], 1),
        ([['Sam', 'u', 'rate'], ['Song1', 's', '_rate'], ['Joey', 'u', 'rate'],['Song3', 's', 'UNK']], 0),
        ([['Sam', 'u', 'rate'], ['Song1', 's', '_rate'], ['Song3', 's', 'UNK']], 0)
    ]

    #For now just construct example, later would want to automatically create maps from vocab
    e_to_ix = {'Sam': 0, 'Weijia': 1, 'Rosa': 2, 'Joey':3, 'Song1': 4, 'Song2': 5, 'Song3': 6, 'Pop': 7}
    t_to_ix = {'u': 0, 's': 1, 't': 2}
    r_to_ix = {'rate': 0, 'category': 1, 'belong': 2, '_rate': 3, '_category': 4, '_belong':5, 'UNK': 6}

    formatted_data = format_paths(training_data, e_to_ix, t_to_ix, r_to_ix)
    print(formatted_data)

    model = train(formatted_data, e_to_ix, t_to_ix, r_to_ix)

    ix_to_etr = construct_ix_to_etr(e_to_ix, t_to_ix, r_to_ix)

    scores_from_paths(model, formatted_data, 3, ix_to_etr)

if __name__ == "__main__":
    main()
