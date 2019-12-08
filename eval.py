import math

def hit_at_k(ranked_tuples, k):
    '''
    Checks if the pos interaction occured in the top k scores
    '''
    for (score, tag) in ranked_tuples[:k]:
        if tag == 1:
            return 1
    return 0

def ndcg_at_k(ranked_tuples, k):
    '''
    If the pos interaction occured in the top k scores,
    return the ndcg relative rank value
    Taken from papers implementation
    '''
    for i,(score, tag) in enumerate(ranked_tuples):
        if tag == 1:
            return math.log(2) / math.log(i + 2)
    return 0
