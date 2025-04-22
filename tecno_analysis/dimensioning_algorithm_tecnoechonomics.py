from model.NetworkGraph import CompleteGraph
from model.NodeClass import Node
from model.LinkClass import Link
import numpy as np
import networkx as nx







class Utility:
    """
    Utility is a class that contains utility functions for the dimensioning algorithm.
    """
    
    @staticmethod
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
    
    



class DimensioningAlgorithmTecnoeconomics:
    """
    DimensioningAlgorithmTecnoeconomics is a class that implements the dimensioning algorithm for the network. TODO: complete
    """
    
    def __init__(
            self,
            node_adjacencies_matrix: np.ndarray,
            node_positions: list[tuple[float, float]],
            node_type_hl4: list[int],   # Quizás no hace falta. Nodos tentativos pueden sustituir
            node_type_hl5: list[int],   # Quizás no hace falta. Nodos tentativos pueden sustituir
            node_traffic_injection: list[float],
            tentative_nodes_for_hpld: list[int],
            tentative_nodes_for_femtocells: list[int],
            tentative_range_for_femtocells: list[float],
            nodes_for_macrocells: list[int],
            range_for_macrocells: list[float],
            euclidean_to_km_scale: float = 1.0,
            power_for_hpld: list[float] = None,
            power_for_femtocells: list[float] = None,
            power_for_macrocells: list[float] = None,
        ):
        """
        Initializes the DimensioningAlgorithmTecnoeconomics class.
        
        This class is responsible for the dimensioning algorithm of the network. It takes in various parameters related to the network topology, node types, traffic injection, and power levels. 
        
        The objective of this class is to determine which nodes are going to be used for HPLD and femtocells, alongside with the association of femtocells to HPLDs.
        
        
        :param node_adjacencies_matrix: The adjacency matrix of the network. NxN matrix where N is the number of nodes.
        :param node_positions: The positions of the nodes in the network. List of N tuples where each tuple contains the x and y coordinates of a node.
        :param node_type_hl4: The types of the nodes in the network. List of N integers, 1 if the node is a HL4 node, 0 otherwise.
        :param node_type_hl5: The types of the nodes in the network. List of N integers, 1 if the node is a HL5 node, 0 otherwise.
        :param node_traffic_injection: The traffic injection for each node in the network. List of N floats where each float represents the traffic injection estimate for a node.
        :param tentative_nodes_for_hpld: The tentative nodes for HPLD. List of N integers, 1 if the node is a tentative node for HPLD, 0 otherwise.
        :param tentative_nodes_for_femtocells: The tentative nodes for femtocells. List of N integers, 1 if the node is a tentative node for femtocells, 0 otherwise.
        :param tentative_range_for_femtocells: The tentative range for femtocells. List of N floats where each float represents the maximum estimated range for a femtocell at node i. The range at index i is the maximum estimated range (km) for the femtocell at node i. Node i must be a tentative node for femtocells.
        :param nodes_for_macrocells: The nodes for macrocells. List of N integers, 1 if the node contains a macrocell, 0 otherwise.
        :param range_for_macrocells: The range for macrocells. List of N floats where each float represents the maximum estimated range for a macrocell at node i. The range at index i is the maximum estimated range (km) for the macrocell at node i. Node i must be a node for macrocells.
        :param euclidean_to_km_scale: The scale factor to convert Euclidean distances to kilometers. Represents how many km correspond to each euclidean unit. Default is 1.0.
        :param power_for_hpld: The power for HPLD. List of N floats where each float represents the transmission power for the HPLD at node i. Node i must be a tentative node for HPLD.
        :param power_for_femtocells: The power for femtocells. List of N floats where each float represents the transmission power for the femtocell at node i. Node i must be a tentative node for femtocells.
        :param power_for_macrocells: The power for macrocells. List of N floats where each float represents the transmission power for the macrocell at node i. Node i must be a node for macrocells.
        
        """
        
        # ¿Do some checks?
        
        # Store values
        self.node_adjacencies_matrix = node_adjacencies_matrix
        self.node_adjacencies_matrix_dijkstra = Utility.dijkstra_distances_matrix(node_adjacencies_matrix)
        self.node_positions = node_positions
        self.node_type_hl4 = node_type_hl4
        self.node_type_hl5 = node_type_hl5
        self.node_traffic_injection = node_traffic_injection
        self.tentative_nodes_for_hpld = tentative_nodes_for_hpld
        self.tentative_nodes_for_femtocells = tentative_nodes_for_femtocells
        self.tentative_range_for_femtocells = tentative_range_for_femtocells
        self.nodes_for_macrocells = nodes_for_macrocells
        self.range_for_macrocells = range_for_macrocells
        self.euclidean_to_km_scale = euclidean_to_km_scale
        self.power_for_hpld = power_for_hpld
        self.power_for_femtocells = power_for_femtocells
        self.power_for_macrocells = power_for_macrocells
        self.nodes_with_hpld = np.zeros(len(node_adjacencies_matrix), dtype=int)
        self.nodes_with_femtocells = np.zeros(len(node_adjacencies_matrix), dtype=int)
        self.hpld_to_femtocell_association = np.zeros((len(node_adjacencies_matrix), len(node_adjacencies_matrix)), dtype=int)
        
    def run_algorithm(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        run_algorithm runs the dimensioning algorithm and returns the results.
        
        The objective of this method is to determine which nodes are going to be used for HPLD and femtocells, alongside with the association of femtocells to HPLDs.
        
        :return: A tuple containing three numpy arrays:
            - nodes_with_hpld: A numpy array of shape (N,) where N is the number of nodes. Each element is 1 if the node is going to be used for HPLD, 0 otherwise.
            - nodes_with_femtocells: A numpy array of shape (N,) where N is the number of nodes. Each element is 1 if the node is going to be used for femtocells, 0 otherwise.
            - hpld_to_femtocell_association: A numpy array of shape (N, N) where N is the number of nodes. Rows represent HPLD nodes and columns represent femtocell nodes. Each element is 1 if the HPLD node i is associated with the femtocell node j, 0 otherwise.
        """
        
        
        # Objetive: determine which nodes are going to be used for HPLD and femtocelds, alongside with the association of femtocells to HPLDs.

        # Use local variables to avoid creating an unfinished solution
        nodes_with_hpld = np.zeros(len(self.node_adjacencies_matrix), dtype=int)
        nodes_with_femtocells = np.zeros(len(self.node_adjacencies_matrix), dtype=int)
        hpld_to_femtocell_association = np.zeros((len(self.node_adjacencies_matrix), len(self.node_adjacencies_matrix)), dtype=int)
        
        
        
        
        
        # Save and return the results
        self.nodes_with_hpld = nodes_with_hpld
        self.nodes_with_femtocells = nodes_with_femtocells
        self.hpld_to_femtocell_association = hpld_to_femtocell_association
        return nodes_with_hpld, nodes_with_femtocells, hpld_to_femtocell_association
        
        
        