

def hit_at_k(ranked_tuples, k):
    '''
    Calculates the number of positive hits in the top k recommended items
    '''
    hits = 0
    for (score, tag) in ranked_tuples[:k]:
        if tag == 1:
            hits += 1

    return hits / k
