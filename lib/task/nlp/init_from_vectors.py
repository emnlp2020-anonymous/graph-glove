from warnings import warn

import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import scipy

from ... import check_numpy, GraphEmbedding


def make_graph_from_vectors(X, *, knn_edges, random_edges=0, virtual_vertices=0, deduplicate=True,
                            directed=True, verbose=False, squared=False, GraphEmbeddingClass=GraphEmbedding,
                            **kwargs):
    """
    Creates graph embedding from an object-feature matrix,
    initialize weights with squared euclidian distances |x_i - x_j|^2_2

    The graph consists of three types of edges:
        * knn edges - connecting vertices to their nearest neighbors
        * random edges - connecting random pairs of vertices to get smallworld property
        * edges to virtual_vertices - adds synthetic vertices to task and connect with all other vertices
                                     (init with k-means)

    :param X: task matrix[num_vertors, vector_dim]
    :param knn_edges: connects vertex to this many nearest neighbors
    :param random_edges: adds this many random edges per vertex (long edges for smallworld property)
    :param virtual_vertices: adds this many new vertices connected to all points, initialized as centroids
    :param deduplicate: if enabled(default), removes all duplicate edges
        (e.g. if the edge was first added via :m:, and then added again via :random_rate:
    :param directed: if enabled, treats (i, j) and (j, i) as the same edge
    :param verbose: if enabled, prints progress into stdout
    :param squared: if True, uses squared euclidian distance, otherwise normal euclidian distance
    :param kwargs: other keyword args sent to :GraphEmbedding.__init__:
    :rtype: GraphEmbedding
    """
    num_vectors, vector_dim = X.shape
    X = np.require(check_numpy(X), dtype=np.float32, requirements=['C_CONTIGUOUS'])
    if virtual_vertices != 0:
        if verbose: print("Creating virtual vertices by k-means")
        X_clusters = KMeans(virtual_vertices).fit(X).cluster_centers_
        X = np.concatenate([X, X_clusters])

    if verbose:
        print("Searching for nearest neighbors")
    try:
        from faiss import IndexFlatL2
        index = IndexFlatL2(vector_dim)
        index.add(X)
        neighbor_distances, neighbor_indices = index.search(X, knn_edges + 1)
    except ImportError:
        warn("faiss not found, using slow knn instead")
        neighbor_distances, neighbor_indices = NearestNeighbors(n_neighbors=knn_edges + 1).fit(X).kneighbors(X)

    if not squared:
        neighbor_distances **= 0.5
    if verbose:
        print("Adding knn edges")
    edges_from, edges_to, distances = [], [], []
    for vertex_i in np.arange(num_vectors):
        for neighbor_i, distance in zip(neighbor_indices[vertex_i], neighbor_distances[vertex_i]):
            if vertex_i == neighbor_i: continue  # forbid loops
            if neighbor_i == -1: continue  # ANN engine uses -1 for padding
            edges_from.append(vertex_i)
            edges_to.append(neighbor_i)
            distances.append(distance)

    if random_edges != 0:
        if verbose: print("Adding random edges")
        random_from = np.random.randint(0, num_vectors, num_vectors * random_edges)
        random_to = np.random.randint(0, num_vectors, num_vectors * random_edges)
        for vertex_i, neighbor_i in zip(random_from, random_to):
            if vertex_i != neighbor_i:
                distance = np.sum((X[vertex_i] - X[neighbor_i]) ** 2)
                if not squared: distance **= 0.5
                edges_from.append(vertex_i)
                edges_to.append(neighbor_i)
                distances.append(distance)

    if deduplicate:
        if verbose: print("Deduplicating edges")
        unique_edges_dict = {}  # {(from_i, to_i) : distance(i, j)}
        for from_i, to_i, distance in zip(edges_from, edges_to, distances):
            edge_iijj = int(from_i), int(to_i)
            if not directed:
                edge_iijj = tuple(sorted(edge_iijj))
            unique_edges_dict[edge_iijj] = distance

        edges_iijj, distances = zip(*unique_edges_dict.items())
        edges_from, edges_to = zip(*edges_iijj)

    edges_from, edges_to, distances = map(np.asanyarray, [edges_from, edges_to, distances])
    if verbose:
        print("Total edges: {}, mean edges per vertex: {}, mean distance: {}".format(
            len(edges_from), len(edges_from) / float(num_vectors), np.mean(distances)
        ))
    return GraphEmbeddingClass(edges_from, edges_to, initial_weights=distances, directed=directed, **kwargs)


def make_graph_from_cooc(cooc_matrix, *, knn_edges, random_edges=0, verbose=False, GraphEmbeddingClass=GraphEmbedding, cooc_mode='DISTANCE',
                         **kwargs):
    """
    :param cooc_matrix: word co-occurrence matrix[n_objects, n_objects]
    :param knn_edges: connects vertex to this many nearest neighbors
    :param random_edges: adds this many random edges per vertex (long edges for smallworld property)
    :param verbose: if enabled, prints progress into stdout
    :param kwargs: other keyword args sent to :GraphEmbedding.__init__:
    :rtype: GraphEmbedding
    """
    assert isinstance(cooc_matrix, scipy.sparse.csr_matrix)
    if verbose:
        print("Adding knn edges")

    if random_edges != 0 and verbose:
        print("Adding random edges")

    num_vertices = cooc_matrix.shape[0]  # zero vertex will have this index
    frequencies = np.asarray(cooc_matrix.sum(1)).flatten()
    edges = {}
    for i, row in enumerate(cooc_matrix):
        col, row_nz = row.nonzero()
        nz_argsort = (-row.data).argsort()

        nearest_inds = nz_argsort[:knn_edges]

        for ind in nearest_inds:
            vertex_from = i
            vertex_to = row_nz[ind]
            if cooc_mode == 'DOT_PRODUCT':
                dist = np.sqrt(2 * np.log(frequencies[vertex_from] * frequencies[vertex_to] / row.data[ind]))
            elif cooc_mode == 'DISTANCE':
                dist = np.log(frequencies[vertex_from] * frequencies[vertex_to] / row.data[ind])
            elif cooc_mode == 'DISTANCE_SQUARED':
                dist = np.sqrt(np.log(frequencies[vertex_from] * frequencies[vertex_to] / row.data[ind]))
            else:
                raise ValueError
            edges[tuple(sorted((i, vertex_to)))] = dist

        if len(row_nz) > knn_edges:
            ranks = np.empty_like(row_nz)
            ranks[nz_argsort] = np.arange(1, len(row_nz) + 1)
            probs = (1 / ranks)
            probs[nearest_inds] = 0
            probs /= probs.sum()

            sampled_neighbors = np.random.choice(len(row_nz), size=min(random_edges, np.count_nonzero(probs)),
                                                 replace=False, p=probs)
            for ind in sampled_neighbors:
                vertex_from = i
                vertex_to = row_nz[ind]
                if cooc_mode == 'DOT_PRODUCT':
                    dist = np.sqrt(2 * np.log(frequencies[vertex_from] * frequencies[vertex_to] / row.data[ind]))
                elif cooc_mode == 'DISTANCE':
                    dist = np.log(frequencies[vertex_from] * frequencies[vertex_to] / row.data[ind])
                elif cooc_mode == 'DISTANCE_SQUARED':
                    dist = np.sqrt(np.log(frequencies[vertex_from] * frequencies[vertex_to] / row.data[ind]))
                else:
                    raise ValueError
                edges[tuple(sorted((i, vertex_to)))] = dist
        if cooc_mode == 'DOT_PRODUCT':
            edges[(i, num_vertices)] = np.sqrt(2 * np.log(frequencies[i]))

    edges_from = np.empty(len(edges), dtype=np.uint32)
    edges_to = np.empty(len(edges), dtype=np.uint32)
    distances = np.empty(len(edges), dtype=np.float32)

    for i, ((source, target), distance) in enumerate(edges.items()):
        edges_from[i] = source
        edges_to[i] = target
        distances[i] = distance

    if verbose:
        print("Total edges: {}, mean edges per vertex: {}, mean distance: {}".format(
            len(edges_from), len(edges_from) / float(cooc_matrix.shape[0]), np.mean(distances)
        ))
    return GraphEmbeddingClass(edges_from, edges_to, initial_weights=distances, directed=False, **kwargs)
