from model.LinkClass import Link
from model.NodeClass import Node
import networkx as nx
import numpy as np
import pandas as pd
import scipy.io
from pydantic import BaseModel
from scipy.spatial import KDTree
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import MDS

def mds_layout(matrix: np.ndarray):
    mds = MDS(dissimilarity="precomputed", random_state=42)
    pos = mds.fit_transform(matrix)
    return pos

graph_functions = {
    "spring_layout": lambda G: nx.spring_layout(G, seed=42),
    "circular_layout": lambda G: nx.circular_layout(G),             # TODO: tune
    "kamada_kawai_layout": lambda G: nx.kamada_kawai_layout(G),     # TODO: tune
    "random_layout": lambda G: nx.random_layout(G),                 # TODO: tune
    "shell_layout": lambda G: nx.shell_layout(G),                   # TODO: tune
}
    
class CompleteGraph(BaseModel):
    """
    CompleteGraph is a class that represents the network of a PoF system. It contains the nodes and links of the network, and subyacent data.
    """
    
    nodes: list[Node]
    links: list[Link]
    nodes_to_discard: list[int]
    links_to_discard: list[int]
    scale_factor: float | None = None
    network_polygon_bounds: list[tuple[float, float]] | None = None
    adjacency_matrix: list[list[float]] | None = None
    
    # ---------------------------------------------------------------------------------------------------------------- #
    # -- Wrapper to create from files -------------------------------------------------------------------------------- #
    #                                                                                                                  #
    # ---------------------------------------------------------------------------------------------------------------- #
    @staticmethod
    def of_sources(distance_matrix_path: str, xlsx_data_path: str, layout_function: str = "spring_layout"):
        """
        Create a CompleteGraph from a distance matrix and an Excel file.
        :param distance_matrix_path (str): Path to the distance matrix file.
        :param xlsx_data_path (str): Path to the Excel file.
        :param layout_function (str, optional): Layout function to use. Defaults to "spring_layout".
        :return: CompleteGraph object.
        """
        distance_matrix = scipy.io.loadmat(distance_matrix_path)['crossMatrix']
        xlsx_data = pd.read_excel(xlsx_data_path)
        
        
        graph = nx.Graph()
        nodes = []
        links = []
        nodes_to_discard = []
        links_to_discard = []
        
        
        num_nodes = distance_matrix.shape[0]
        map_id_to_obj = {}
        adjacency_matrix = np.zeros((num_nodes, num_nodes))

        for i in range(num_nodes):
            if xlsx_data.iloc[i, 1] != "HL4" and xlsx_data.iloc[i, 1] != "HL5":
                nodes_to_discard.append(i)


        for i in range(num_nodes):
            if i not in nodes_to_discard:
                graph.add_node(i)
        
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i == j or i in nodes_to_discard or j in nodes_to_discard:
                    continue
                distance = distance_matrix[i, j]
                adjacency_matrix[i, j] = distance
                
                if distance > 0:
                    graph.add_edge(i, j, weight=1/distance)

        # Compute layout positions
        if layout_function == "mds_layout":
            pos = mds_layout(distance_matrix)
        else:
            pos = graph_functions[layout_function](graph)

        # Create nodes
        for i in range(num_nodes):
            if i not in nodes_to_discard:
                node_name = xlsx_data.iloc[i, 0]
                node_type = xlsx_data.iloc[i, 1]
                node_degree = int(sum(distance_matrix[i] > 0))
                node_x, node_y = pos[i]

                nodes.append(Node(name=node_name, pos=(node_x, node_y), node_degree=node_degree, type=node_type))
                map_id_to_obj[i] = len(nodes) - 1
        
        # Create links
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                if i == j or i in nodes_to_discard or j in nodes_to_discard:
                    continue
                distance = distance_matrix[i, j]
                id_a = map_id_to_obj[i]
                id_b = map_id_to_obj[j]
                
                if distance > 0:
                    node_a = nodes[id_a]
                    node_b = nodes[id_b]
                    newLink = Link(a=node_a, b=node_b, distance_km=distance, label=f'{distance:.2f}km', name=f'{node_a.name} <-> {node_b.name}')
                    links.append(newLink)

                    node_a.assoc_links.append(newLink.name)
                    node_b.assoc_links.append(newLink.name)
                    node_a.assoc_nodes.append(node_b.name)
                    node_b.assoc_nodes.append(node_a.name)

        adjacency_matrix = [[float(adjacency_matrix[i][j]) for j in range(len(adjacency_matrix[0]))] for i in range(len(adjacency_matrix))]
        
        return CompleteGraph(nodes=nodes, links=links, nodes_to_discard=nodes_to_discard, links_to_discard=links_to_discard, adjacency_matrix=adjacency_matrix)
      

      
      
      
    @staticmethod
    def of_sources_only_core_nodes(
        distance_matrix_path: str,
        xlsx_data_path: str,
        layout_function: str = "spring_layout",
        node_types_to_discard: list[str] = ["HL4", "HL5"],
        core_nodes: list[str] = None,
    ):
        """
        Create a CompleteGraph from a distance matrix and an Excel file.
        
        Args:
            distance_matrix_path (str): Path to the .mat file containing the distance matrix
            xlsx_data_path (str): Path to the Excel file containing node information
            layout_function (str, optional): Name of the layout function to use. Defaults to "spring_layout"
            node_types_to_discard (list[str], optional): List of node types to exclude. Defaults to ["HL4", "HL5"]
            
        Returns:
            CompleteGraph: A CompleteGraph object containing only core nodes (excluding specified node types)
            
        The distance matrix file should be a .mat file containing a matrix named 'crossMatrix' with distances between nodes.
        The Excel file should contain node names in the first column and node types in the second column.
        """
        distance_matrix = scipy.io.loadmat(distance_matrix_path)['crossMatrix']
        xlsx_data = pd.read_excel(xlsx_data_path)
        
        
        
        # If core nodes, only allow nodes and links directly connected to core nodes, thus, discarding all others
        
        
        
        graph = nx.Graph()
        nodes = []
        links = []
        nodes_to_discard = []
        links_to_discard = []

        
        banned_links = [] # Tuple (node, node)
        
        
        num_nodes = distance_matrix.shape[0]
        map_id_to_obj = {}
        adjacency_matrix = np.zeros((num_nodes, num_nodes))

        # Add nodes to discard based on node type
        for i in range(num_nodes):
            if xlsx_data.iloc[i, 1] in node_types_to_discard:
                nodes_to_discard.append(i)
                
                
                
        # If core nodes, only allow nodes and links directly connected to core nodes
        if core_nodes is not None:
            core_nodes_ids = []

            # Find indices of core nodes
            for i in range(num_nodes):
                name = xlsx_data.iloc[i, 0]
                if name in core_nodes:
                    core_nodes_ids.append(i)
                    
            
            # Find adjacencies to core nodes, discard elsewhere
            for node in range(num_nodes):
                if node in core_nodes_ids:
                    continue
                
                # Check if the node is connected to any core node
                is_connected_to_core = False
                for core_node in core_nodes_ids:
                    if distance_matrix[node, core_node] > 0:
                        is_connected_to_core = True
                        break
                
                if not is_connected_to_core:
                    nodes_to_discard.append(node)
                    
            # Find non-core nodes with more than 1 link
            for node in range(num_nodes):
                if node in core_nodes_ids or node in nodes_to_discard:
                    continue
                
                # Count links to core nodes and track connections
                connected_to = []
                for core_node in core_nodes_ids:
                    if distance_matrix[node, core_node] > 0:
                        connected_to.append(core_node)
                
                # If connected to more than one core node, discard links until only one remains
                if len(connected_to) > 1:
                    # Sort connections by distance to prioritize closer links
                    connected_to.sort(key=lambda core_node: distance_matrix[node, core_node])
                    
                    # Keep only the closest link and discard the rest
                    for core_node in connected_to[1:]:
                        banned_links.append((node, core_node))
                
                
        

        # Only add non-discarded nodes to graph
        for i in range(num_nodes):
            if i not in nodes_to_discard:
                graph.add_node(i)
        
        # Only add non-discarded links to graph
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i == j or i in nodes_to_discard or j in nodes_to_discard:
                    continue

                
                if core_nodes is not None and (i not in core_nodes_ids and j not in core_nodes_ids):
                    continue
                
                if (i, j) in banned_links or (j, i) in banned_links:
                    continue
                
                distance = distance_matrix[i, j]
                adjacency_matrix[i, j] = distance
                
                if distance > 0:
                    graph.add_edge(i, j, weight=1/distance)

        # Compute layout positions
        if layout_function == "mds_layout":
            pos = mds_layout(distance_matrix)
        else:
            pos = graph_functions[layout_function](graph)

        # Create nodes
        for i in range(num_nodes):
            if i not in nodes_to_discard:
                node_name = xlsx_data.iloc[i, 0]
                node_type = xlsx_data.iloc[i, 1]
                node_degree = int(sum(distance_matrix[i] > 0))
                node_x, node_y = pos[i]

                nodes.append(Node(name=node_name, pos=(node_x, node_y), node_degree=node_degree, type=node_type))
                map_id_to_obj[i] = len(nodes) - 1
        
        # Create links
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                if i == j or i in nodes_to_discard or j in nodes_to_discard:
                    continue
                
                if core_nodes is not None and (i not in core_nodes_ids and j not in core_nodes_ids):
                    continue
                
                if (i, j) in banned_links or (j, i) in banned_links:
                    continue
                
                distance = distance_matrix[i, j]
                id_a = map_id_to_obj[i]
                id_b = map_id_to_obj[j]
                
                if distance > 0:
                    node_a = nodes[id_a]
                    node_b = nodes[id_b]
                    newLink = Link(a=node_a, b=node_b, distance_km=distance, label=f'{distance:.2f}km', name=f'{node_a.name} <-> {node_b.name}')
                    links.append(newLink)

                    node_a.assoc_links.append(newLink.name)
                    node_b.assoc_links.append(newLink.name)
                    node_a.assoc_nodes.append(node_b.name)
                    node_b.assoc_nodes.append(node_a.name)

        adjacency_matrix = [[float(adjacency_matrix[i][j]) for j in range(len(adjacency_matrix[0]))] for i in range(len(adjacency_matrix))]
        
        return CompleteGraph(nodes=nodes, links=links, nodes_to_discard=nodes_to_discard, links_to_discard=links_to_discard, adjacency_matrix=adjacency_matrix)
      
   
                    
                
    
    # ---------------------------------------------------------------------------------------------------------------- #
    # -- Constructor ------------------------------------------------------------------------------------------------- #
    #                                                                                                                  #
    # ---------------------------------------------------------------------------------------------------------------- #
    def __init__(self, 
        nodes: list[Node],
        links: list[Link],
        nodes_to_discard: list[int],
        links_to_discard: list[int],
        adjacency_matrix: list[list[float]],
    ):
        """
        Constructor for the CompleteGraph class.
        :param nodes: List of nodes.
        :param links: List of links.
        :param nodes_to_discard: List of nodes to discard.
        :param links_to_discard: List of links to discard.
        :return: CompleteGraph object.
        """
        super().__init__(nodes=nodes, links=links, nodes_to_discard=nodes_to_discard, links_to_discard=links_to_discard, adjacency_matrix=adjacency_matrix)
                    
        print(f'Discarded nodes: {len(self.nodes_to_discard)}')
        print(f'Discarded links: {len(self.links_to_discard)}')
        
        # print(f'Vertexs: {len(self.vertexs)}')
        # print(f'Edges: {len(self.edges)}')
        print(f'Links: {len(self.links)}')
        print(f'Nodes: {len(self.nodes)}')
        
        self.compute_traffic_profiles()
        self.print_network()
        self.transform_nodes_coordinates()
        self.translate_coordinates_to_possitive_values()
    
    # ---------------------------------------------------------------------------------------------------------------- #
    # -- Compute traffic profiles ------------------------------------------------------------------------------------ #
    #                                                                                                                  #
    # Inputs:                                                                                                          #
    # - radio: radio de la ventana de vecinos                                                                          #
    # - alpha: coeficiente de ponderación de la densidad                                                               #
    # - beta: coeficiente de ponderación del grado                                                                     #
    # - traffic_profile_threshold_low: umbral de intensidad para el perfil de tráfico bajo                             #
    # - traffic_profile_threshold_medium: umbral de intensidad para el perfil de tráfico medio                         #
    # - traffic_profile_threshold_high: umbral de intensidad para el perfil de tráfico alto                            #
    # - traffic_profile_mbps_low: intensidad de tráfico en Mbps para el perfil de tráfico bajo                         #
    # - traffic_profile_mbps_medium: intensidad de tráfico en Mbps para el perfil de tráfico medio                     #
    # - traffic_profile_mbps_high: intensidad de tráfico en Mbps para el perfil de tráfico alto                        #
    # ---------------------------------------------------------------------------------------------------------------- #
    def compute_traffic_profiles(
            self, 
            radio: float = 0.5, 
            alpha: float = 1.0, 
            beta: float = 1.0,
            traffic_profile_threshold_low: float = 0.4,
            traffic_profile_threshold_medium: float = 0.7,
            traffic_profile_threshold_high: float = 1.0,
            traffic_profile_mbps_low: float = 250,
            traffic_profile_mbps_medium: float = 500,
            traffic_profile_mbps_high: float = 1000,
        ):
        """
        Compute traffic profiles estimation for each node in the network, based on the density and degree of the nodes in a neighborhood window.
        
        The density is computed as the number of nodes in a neighborhood window of radius `radio`.
        The degree is the number of edges connected to the node.
        The intensity is computed as a linear combination of the density and the degree, weighted by `alpha` and `beta` respectively.
        The intensity is then normalized to the range [0, 1] and classified into one of the three traffic profiles: low, medium, or high.
        
        The traffic profiles are used to estimate the traffic injection for each node in the network.
        
        Args:
            radio (float, optional): Radio of the neighborhood window. Defaults to 0.5.
            alpha (float, optional): Weighting coefficient for density. Defaults to 1.0.
            beta (float, optional): Weighting coefficient for degree. Defaults to 1.0.
            traffic_profile_threshold_low (float, optional): Threshold for low traffic profile. Defaults to 0.4.
            traffic_profile_threshold_medium (float, optional): Threshold for medium traffic profile. Defaults to 0.7.
            traffic_profile_threshold_high (float, optional): Threshold for high traffic profile. Defaults to 1.0.
            traffic_profile_mbps_low (float, optional): Traffic intensity in Mbps for low traffic profile. Defaults to 250.
            traffic_profile_mbps_medium (float, optional): Traffic intensity in Mbps for medium traffic profile. Defaults to 500.
            traffic_profile_mbps_high (float, optional): Traffic intensity in Mbps for high traffic profile. Defaults to 1000.
        """
        
        # Get map coordinates and degree
        # x, y = Vertex.obtain_x_y_vectors(self.vertexs)
        x, y = Node.obtain_x_y_vectors(self.nodes)
        # degree = Vertex.obtain_degree_vector(self.vertexs)
        degree = Node.obtain_degree_vector(self.nodes)
        df = pd.DataFrame({'x': x, 'y': y, 'degree': degree})

        # Calcular densidad local: cuántos vecinos en cierto radio
        tree = KDTree(df[['x', 'y']])
        densidad = []
        
        for i in range(len(df)):
            vecinos = tree.query_ball_point([df.loc[i, 'x'], df.loc[i, 'y']], r=radio)
            densidad.append(len(vecinos) - 1)  # excluye el propio nodo
            
        df['densidad'] = densidad

        # Estimar intensidad de tráfico
        df['intensidad'] = alpha * df['densidad'] + beta * df['degree']
        
        # Normalizar para visualización
        scaler = MinMaxScaler()
        df['intensidad_norm'] = scaler.fit_transform(df[['intensidad']])
        
        
        
        # Clasificar cada nodo según su intensidad
        # [0, 0.4) -> 0
        # [0.4, 0.7) -> 1
        # [0.7, 1.0] -> 2
        
        for i in range(len(df)):
            if df.loc[i, 'intensidad_norm'] < traffic_profile_threshold_low:
                self.nodes[i].traffic_profile = "low"
                self.nodes[i].estimated_traffic_injection = traffic_profile_mbps_low
            elif df.loc[i, 'intensidad_norm'] < traffic_profile_threshold_medium:
                self.nodes[i].traffic_profile = "medium"
                self.nodes[i].estimated_traffic_injection = traffic_profile_mbps_medium
            elif df.loc[i, 'intensidad_norm'] <= traffic_profile_threshold_high:
                self.nodes[i].traffic_profile = "high"
                self.nodes[i].estimated_traffic_injection = traffic_profile_mbps_high
                
                
    
    # ---------------------------------------------------------------------------------------------------------------- #
    # -- Transform nodes coordinates from normalized to scaled by the distance in km --------------------------------- #
    #                                                                                                                  #
    # Find the largest distance link, find both ends of the link, and scale the coordinates of the nodes to the        #
    # distance in km.                                                                                                  #
    # ---------------------------------------------------------------------------------------------------------------- #
    def transform_nodes_coordinates(self):
        """
        Transform nodes coordinates from normalized to scaled by the distance in km.
        """
        # Find the largest distance link (best precission)
        scale_factors = []
        for link in self.links:
            node_a = link.a
            node_b = link.b
            norm_distance = np.sqrt((node_a.pos[0] - node_b.pos[0])**2 + (node_a.pos[1] - node_b.pos[1])**2)
            scale_factors.append(link.distance_km / norm_distance)
            
        scale_factor = np.mean(scale_factors)
                
        # Scale the coordinates of each node
        for node in self.nodes:
            node.pos = (node.pos[0] * scale_factor, node.pos[1] * scale_factor)
            
            
        # Find the network bounds
        margin = 2
        min_x = min([node.pos[0] for node in self.nodes]) - margin
        max_x = max([node.pos[0] for node in self.nodes]) + margin
        min_y = min([node.pos[1] for node in self.nodes]) - margin
        max_y = max([node.pos[1] for node in self.nodes]) + margin
        self.network_polygon_bounds = [(min_x, min_y), (min_x, max_y), (max_x, max_y), (max_x, min_y), (min_x, min_y)]
        self.scale_factor = scale_factor

    
    # ---------------------------------------------------------------------------------------------------------------- #
    # -- Translate coordinates to possitive values, thus, the first quadrant ----------------------------------------- #
    #                                                                                                                  #
    # ---------------------------------------------------------------------------------------------------------------- #
    def translate_coordinates_to_possitive_values(self):
        """
        Translate coordinates to possitive values, thus, the first quadrant.
        """
        min_x = min([node.pos[0] for node in self.nodes])
        min_y = min([node.pos[1] for node in self.nodes])
        
        margin = 2


        if min_x > margin:
            min_x = 0 # avoid translating to the left
        if min_y > margin:
            min_y = 0 # avoid translating to the bottom
        
        for node in self.nodes:
            node.pos = (node.pos[0] - min_x + margin, node.pos[1] - min_y + margin)
            
        # Find the network bounds
        min_x = min([node.pos[0] for node in self.nodes]) - margin
        max_x = max([node.pos[0] for node in self.nodes]) + margin
        min_y = min([node.pos[1] for node in self.nodes]) - margin
        max_y = max([node.pos[1] for node in self.nodes]) + margin
        self.network_polygon_bounds = [(min_x, min_y), (min_x, max_y), (max_x, max_y), (max_x, min_y), (min_x, min_y)]
      
            
            
    
    # ---------------------------------------------------------------------------------------------------------------- #
    # -- Calculate adjacency matrix ---------------------------------------------------------------------------------- #
    #                                                                                                                  #
    # ---------------------------------------------------------------------------------------------------------------- #
    def calc_adjacency_matrix(self) -> np.ndarray:
        """
        Returns the adjacency matrix of the graph.
        :return: Adjacency matrix of the graph.
        """
        
        adj_matrix = np.zeros((len(self.nodes), len(self.nodes)))
        for link in self.links:
            idx_a = self.nodes.index(link.a)
            idx_b = self.nodes.index(link.b)
            adj_matrix[idx_a, idx_b] = link.distance_km
            adj_matrix[idx_b, idx_a] = link.distance_km
        
        return adj_matrix
            
            
            
            
    
    # ---------------------------------------------------------------------------------------------------------------- #
    # -- Print network ----------------------------------------------------------------------------------------------- #
    # ---------------------------------------------------------------------------------------------------------------- #
    def print_network(self):
        print('\n*-*-* Printing information about the imported network *-*-*\n')
        # Num nodes
        print(f'Num nodes: {len(self.nodes)}')
        # Num links
        print(f'Num links: {len(self.links)}')
        # Num HL4
        print(f'Num HL4: {len([node for node in self.nodes if node.type == "HL4"])}')
        # Num HL5
        print(f'Num HL5: {len([node for node in self.nodes if node.type == "HL5"])}')
        
        # Average distance
        print(f'Average distance: {np.mean([link.distance_km for link in self.links]):.2f}')
        # Max distance
        print(f'Max distance (km): {np.max([link.distance_km for link in self.links]):.2f}')
        # Min distance
        print(f'Min distance (km): {np.min([link.distance_km for link in self.links]):.2f}')
        
        # Average degree
        print(f'Average degree: {np.mean([node.node_degree for node in self.nodes]):.2f}')
        # Min degree
        print(f'Min degree: {np.min([node.node_degree for node in self.nodes])}')
        # Max degree
        print(f'Max degree: {np.max([node.node_degree for node in self.nodes])}')
        
        # Average degree HL4
        print(f'Average degree HL4: {np.mean([node.node_degree for node in self.nodes if node.type == "HL4"]):.2f}')
        # Average degree HL5
        print(f'Average degree HL5: {np.mean([node.node_degree for node in self.nodes if node.type == "HL5"]):.2f}')
        
        # Total link length
        print(f'Total bidirectional link length (km): {np.sum([link.distance_km for link in self.links]):.2f}')
        # print(f'Total directional link length (km): {2 * np.sum([edge.distance for edge in self.edges]):.2f}')
        
        
        
        # Num links HL4 - HL5
        # Num links HL5 - HL5
        # Num links HL4 - HL4
        
        # Is connex graph
    
