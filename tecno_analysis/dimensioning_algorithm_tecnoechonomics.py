import numpy as np
import tecno_analysis.utility as Utils



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
            fixed_nodes_for_hpld: list[int],
            tentative_nodes_for_hpld: list[int],
            tentative_nodes_for_femtocells: list[int],
            tentative_range_for_femtocells: list[float],
            nodes_for_macrocells: list[int],
            range_for_macrocells: list[float],
            max_runtime_seconds: float = 180.0, # 3 minutes
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
        :param fixed_nodes_for_hpld: The fixed nodes for HPLD. List of N integers, 1 if the node has an HPLD, 0 otherwise.
        :param tentative_nodes_for_hpld: The tentative nodes for HPLD. List of N integers, 1 if the node is a tentative to have HPLD, 0 otherwise.
        :param tentative_nodes_for_femtocells: The tentative nodes for femtocells. List of N integers, 1 if the node is a tentative to have a femtocell, 0 otherwise.
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
        self.node_adjacencies_matrix_dijkstra = Utils.dijkstra_distances_matrix(node_adjacencies_matrix)
        self.node_positions = node_positions
        self.node_type_hl4 = node_type_hl4
        self.node_type_hl5 = node_type_hl5
        self.node_traffic_injection = node_traffic_injection
        self.fixed_nodes_for_hpld = fixed_nodes_for_hpld
        self.tentative_nodes_for_hpld = tentative_nodes_for_hpld
        self.tentative_nodes_for_femtocells = tentative_nodes_for_femtocells
        self.tentative_range_for_femtocells = tentative_range_for_femtocells
        self.nodes_for_macrocells = nodes_for_macrocells
        self.range_for_macrocells = range_for_macrocells
        self.max_runtime_seconds = max_runtime_seconds
        self.euclidean_to_km_scale = euclidean_to_km_scale
        self.power_for_hpld = power_for_hpld
        self.power_for_femtocells = power_for_femtocells
        self.power_for_macrocells = power_for_macrocells
        
        # Solution variables
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
        
        
        
        # Overview of the algorithm:
        # 1. Dimension the location of the femtocells. Most costly part of the algorithm (computational cost of objective function).
        # 2. Dimension the location of the HPLDs and associate them to the femtocells.
        
        
        # 1. Dimension the location of the femtocells
        
        
        
        # 2. Dimension the location of the HPLDs and associate them to the femtocells.
        
        
        
        # Save and return the results
        self.nodes_with_hpld = nodes_with_hpld
        self.nodes_with_femtocells = nodes_with_femtocells
        self.hpld_to_femtocell_association = hpld_to_femtocell_association
        return nodes_with_hpld, nodes_with_femtocells, hpld_to_femtocell_association
        
        
        
    
    
    
    
    
    
    
    
    
    
    # ---------------------------------------------------------------------------------------------------------------- #
    # -- Objective cost functions ------------------------------------------------------------------------------------ #
    #                                                                                                                  #
    #                                                                                                                  #
    # ---------------------------------------------------------------------------------------------------------------- #

    
    def cost_function_hpld(self):
        pass
    def augmented_cost_function_hpld(self):
        pass
    
    
    
    
    
    # ---------------------------------------------------------------------------------------------------------------- #
    # -- Auxiliary functions ----------------------------------------------------------------------------------------- #
    #                                                                                                                  #
    # Auxiliary functions to calculate parameters from the network graph.                                              #
    # ---------------------------------------------------------------------------------------------------------------- #
    
    @staticmethod
    def num_associated_femtos_to_hpld(association_matrix:np.ndarray, hpld_index: int):
        pass
    
    
    
    
    
    



    
def cost_function_femtocell(self):
    pass

def augmented_cost_function_femtocell(self):
    pass

    

def determine_best_femtos(
    node_positions: list[tuple[float, float]],
    tentative_nodes_for_femtocells: list[int],
    tentative_range_for_femtocells: list[float],
    traffic_injection: list[float],
    base_area: list[tuple[float, float]],
) -> list[int]:
    num_tentative_femtos = sum(tentative_nodes_for_femtocells)
    num_nodes = len(node_positions)
    
    initial_solution = np.zeros(len(node_positions), dtype=int)
    
    # Random initial solution
    num_femtos = np.random.randint(tentative_nodes_for_femtocells/4, tentative_nodes_for_femtocells)
    random_nodes = np.random.choice(num_nodes, num_femtos, replace=False)
    
    initial_solution[random_nodes] = 1
    
    
    # 
    
    num_loops_no_improvement = 0
    num_loops_max = 100
    best_solution = initial_solution
    best_cost = cost_function_femtocell(initial_solution, traffic_injection)
    
    while num_loops_no_improvement < num_loops_max:
        
        # this_loop_solution = best_solution.copy()
        this_loop_cost = best_cost
        
        for node in range(num_nodes):
            local_solution = initial_solution.copy()
            
            # Flip the node state
            if local_solution[node] == 1:
                local_solution[node] = 0
            else:
                local_solution[node] = 1
                
            # Calculate the cost of the new solution
            new_cost = cost_function_femtocell(local_solution, traffic_injection)
            
            # Check if the new solution is better
            if new_cost < best_cost:
                best_cost = new_cost
                best_solution = local_solution
                
        
        if best_cost < this_loop_cost:      # If the cost has improved, reset the counter
            num_loops_no_improvement = 0
        else:   
            num_loops_no_improvement += 1
                
            
        
    
    