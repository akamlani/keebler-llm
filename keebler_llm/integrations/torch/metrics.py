import torch 
import torch.nn.functional as F

get_euclidean_dist = lambda x, y: F.pairwise_distance(x, y, p=2)
get_manhattan_dist = lambda x, y: F.pairwise_distance(x, y, p=1)
get_cosine_dist    = lambda x, y: 1-F.cosine_similarity(x, y)
