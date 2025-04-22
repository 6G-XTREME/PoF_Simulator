import numpy as np
import networkx as nx


def dijkstra_distances_matrix(adjacencies_matrix: np.ndarray) -> np.ndarray:
    """
    dijkstra_distances_matrix calculates the shortest path distances between all pairs of nodes in the network using Dijkstra's algorithm.
    
    :param adjacencies_matrix: The adjacency matrix of the network.
    :return: A matrix containing the shortest path distances between all pairs of nodes.
    """
    graph = nx.from_numpy_array(adjacencies_matrix)
    raw_dijkstra_matrix = dict(nx.all_pairs_dijkstra_path_length(graph))
    # return np.ndarray(shape=(len(raw_dijstra_matrix), len(raw_dijstra_matrix)), dtype=float, order='C', buffer=nx.to_numpy_array(graph, weight='weight'))
    
    num_nodes = len(raw_dijkstra_matrix)
    # shortest_distances = np.full((num_nodes, num_nodes), float('inf'))  # Initialize with infinity
    shortest_distances = np.full((num_nodes, num_nodes), 0)  # Initialize with infinity

    # Fill the shortest distances into the matrix
    for i in range(num_nodes):
        for j in range(num_nodes):
            # Check if there's a path from node i to node j
            if j in raw_dijkstra_matrix[i]:
                shortest_distances[i][j] = raw_dijkstra_matrix[i][j]

    return shortest_distances