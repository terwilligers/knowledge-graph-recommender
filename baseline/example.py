import numpy as np
from scipy.io import mmread, mmwrite
from scipy import sparse
from bpr import *

np_matrix = np.random.randint(2, size=(20,30))
scipy_matrix = sparse.csr_matrix(np_matrix)
data = scipy_matrix

args = BPRArgs()
args.learning_rate = 0.01

num_factors = 10
model = BPR(num_factors,args)

sample_negative_items_empirically = True
sampler = UniformPairWithoutReplacement(sample_negative_items_empirically)
num_iters = 500
model.train(data,sampler,num_iters)
