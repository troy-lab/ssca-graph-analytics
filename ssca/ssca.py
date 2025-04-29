from sklearn.neighbors import NearestNeighbors
import numpy as np
from sklearn.metrics import pairwise_distances


def get_k_nearest_neighbors_array(k:int, dataset:np.ndarray) -> np.ndarray:
    """
    Step one sigma value is weighted using k-nearest neighbors.
    We need k+1, because the closest will always be itself.
    :return: list of k nearest neighbors
    """
    neighbors = NearestNeighbors(n_neighbors=k+1, algorithm='auto', metric='euclidean').fit(dataset)
    distances, indices = neighbors.kneighbors(dataset)
    return distances, indices


def calculate_sigmas(distances, k):
    """
    These are the sigma values needed for step 1.
    :param distances:
    :param k:
    :return:
    """
    sigma = distances.sum(axis=1)
    sigma /= k
    return sigma


def calculate_similarity_matrix(dataset, k):
    """
    This is the full step 1.
    :param dataset:
    :param k:
    :return:
    """
    distances, _ = get_k_nearest_neighbors_array(k, dataset)
    sigma = calculate_sigmas(distances, k)
    w_size = dataset.shape[0]
    W = np.zeros((w_size,w_size,))
    sq_euclidean_distances = pairwise_distances(dataset, metric='sqeuclidean') * -1
    for i in range(w_size):
        for j in range(w_size):
            wij = np.exp(sq_euclidean_distances[i,j]/(sigma[i]*sigma[j]))
            W[i,j] = wij
    return W