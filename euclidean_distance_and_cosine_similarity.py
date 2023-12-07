import numpy as np

def euclidean_distance(a, b):
	temp = a - b
	euclid_dist = np.sqrt(np.dot(temp.T, temp))
	return euclid_dist

def cosine_similarity(a, b):
	dot_product = np.dot(a, b)
	norm_a = np.linalg.norm(a)
	norm_b = np.linalg.norm(b)
	return dot_product / (norm_a * norm_b)

def manhattan_distance(a, b):
    return np.sum(np.abs(a - b))
