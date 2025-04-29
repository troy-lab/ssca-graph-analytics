import random
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy import linalg


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
    :param W: Similarity matrix
    :param must_link: required links (same label)
    :param cannot_link: cannot link (different label)
    :return:
    """
    for point in must_link:
        i = point[0]
        j = point[1]
        W[i,j] = 1
        W[j,i] = 1
    for point in cannot_link:
        i = point[0]
        j = point[1]
        W[i,j] = 0.001
        W[j,i] = 0.001
    return W


def construct_L(W):
    """
    This constructs L which we will use to calculate our eigenvalues
    :param W:
    :return:
    """
    d = W.sum(axis=1)
    D_inv_sqrt = np.diag(1 / np.sqrt(d))
    L = D_inv_sqrt.dot(W).dot(D_inv_sqrt)
    return L


def get_eig(matrix):
    """
    This is step 3. Calculate unit orthogonal eigenvectors and eigenvalues
    :param matrix: L
    :return: tuple[array, array]
    """
    eigenvalues, eigenvectors  = linalg.eigh(matrix)
    return eigenvalues, eigenvectors


def sample_links(num_links, labels):
    must_link = []
    cannot_link = []
    for l in list(set(labels)):
        indices = list(np.where(labels == l)[0])
        for _ in range(num_links):
            a, b = random.sample(indices, 2)
            must_link.append((int(a), int(b),))
            not_indices = list(np.where(labels != l)[0])
            a = random.sample(not_indices, 1)[0]
            b = random.sample(indices, 1)[0]
            cannot_link.append((int(a), int(b),))
    return must_link, cannot_link

def ssca(X, y, must_link, cannot_link):
    k = len(list(set(y)))
    W = calculate_similarity_matrix(X, k)
    W = update_pairwise_constraints(W, must_link, cannot_link)
    assert np.allclose(W, W.T)
    L = construct_L(W)
    eigenvalues, eigenvectors = get_eig(L)
    idx = np.argsort(eigenvalues)[-k:]
    idx = idx[np.argsort(eigenvalues[idx])[::-1]]
    X = eigenvectors[:, idx]
    norms = np.linalg.norm(X, axis=1, keepdims=True)   # shape (n,1)
    Y = X / norms
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=100,random_state=0)
    km.fit(Y)
    y_hat = km.labels_
    return y_hat


