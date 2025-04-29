from sklearn.neighbors import NearestNeighbors
import numpy as np
from sklearn.metrics import pairwise_distances


# Step 1 functions - construct similarity matrix.
def get_k_nearest_neighbors_array(k:int, dataset:np.ndarray) -> np.ndarray:
    """
    Step one sigma value is weighted using k-nearest neighbors.
    We need k+1, because the closest will always be itself.
    :return: list of k nearest neighbors
    """
    neighbors = NearestNeighbors(n_neighbors=k+1, algorithm='auto', metric='euclidean').fit(dataset)
    distances, _ = neighbors.kneighbors(dataset)
    return distances


def calculate_sigmas(distances, k):
    """
    These are the sigma values needed for step 1.
    :param distances: euclidean distance matrix
    :param k: number of nearest neighbors
    :return: array of sigma values
    """
    sigma = distances.sum(axis=1)
    sigma /= k
    return sigma


def calculate_similarity_matrix(dataset, k):
    """
    This is the full step 1.
    :param dataset: input dataset (no labels)
    :param k: number of nearest neighbors
    :return: Similarity matrix
    """
    distances = get_k_nearest_neighbors_array(k, dataset)
    sigma = calculate_sigmas(distances, k)
    w_size = dataset.shape[0]
    W = np.zeros((w_size,w_size,))
    sq_euclidean_distances = pairwise_distances(dataset, metric='sqeuclidean') * -1
    eps = 1e-8
    for i in range(w_size):
        for j in range(w_size):
            sigma_i = np.maximum(sigma[i], eps)
            sigma_j = np.maximum(sigma[j], eps)
            wij = np.exp(sq_euclidean_distances[i,j]/(sigma_i*sigma_j))
            W[i,j] = wij
    return W


def update_pairwise_constraints(W, must_link, cannot_link):
    """
    This step adds the link constraints to the similarity matrix
    :param W: similarity matrix
    :param must_link: required links (same label)
    :param cannot_link: cannot link (different label)
    :return:
    """
    for point in must_link:
        i = point[0]
        j = point[1]
        W[i,j] = 0
        W[j,i] = 0
    for point in cannot_link:
        i = point[0]
        j = point[1]
        W[i,j] = 1e6
        W[j,i] = 1e6
    return W