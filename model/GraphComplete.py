from model.LinkClass import Link
from model.NodeClass import Node
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io

class GraphComplete:
    graph_functions = {
        "spring_layout": lambda G: nx.spring_layout(G, seed=42),
        "circular_layout": lambda G: nx.circular_layout(G),
        "kamada_kawai_layout": lambda G: nx.kamada_kawai_layout(G),
        "random_layout": lambda G: nx.random_layout(G),
        "shell_layout": lambda G: nx.shell_layout(G),
    }
    
    @staticmethod
    def of(distance_matrix_path: str, xlsx_data_path: str, layout_function: str = "spring_layout"):
        distance_matrix = scipy.io.loadmat(distance_matrix_path)['crossMatrix']
        xlsx_data = pd.read_excel(xlsx_data_path)
        return GraphComplete(distance_matrix, xlsx_data, layout_function)
    
    
    
    def __init__(self, distance_matrix: np.ndarray, xlsx_data: pd.DataFrame, layout_function: str = "spring_layout"):
        self.graph = nx.Graph()
        self.links = []
        self.nodes = []


        for i in range(distance_matrix.shape[0]):
            self.graph.add_node(i)
        for i in range(distance_matrix.shape[0]):
            for j in range(i + 1, distance_matrix.shape[1]):
                distance = distance_matrix[i, j]
                if distance > 0:
                    self.graph.add_edge(i, j, weight=distance)
                    self.links.append(Link(source=i, target=j, distance=distance, label=f'{distance:.2f}km'))
                    
        self.layout_function = layout_function
        self.pos = self.graph_functions[layout_function](self.graph)
        
        
        for i in range(distance_matrix.shape[0]):
            node_id = i
            node_name = xlsx_data.iloc[i, 0]
            node_type = xlsx_data.iloc[i, 1]
            node_degree = int(sum(distance_matrix[i] > 0))
            node_x, node_y = self.pos[i]
            self.nodes.append(Node(id=node_id, type=node_type, x=node_x, y=node_y, node_degree=node_degree, name=node_name))
        
        
    
    
    def plot_graph(self, guardar_figura=True, nombre_figura="grafo_distancias.png"):
        fig, ax = plt.subplots(figsize=(40, 40))
        node_colors = ["yellow" if node.type == "HL4" else "green" if node.type == "HL5" else "blue" for node in self.nodes]
        nodes = nx.draw_networkx_nodes(self.graph, self.pos, node_size=100, node_color=node_colors, cmap=plt.cm.viridis, ax=ax)
        nx.draw_networkx_edges(self.graph, self.pos, alpha=0.3, ax=ax)

        # Mostrar etiquetas de nodos si hay pocos
        if len(self.graph.nodes) <= 200:
            nx.draw_networkx_labels(self.graph, self.pos, {i: f"{self.nodes[i].name}" for i in self.graph.nodes()}, font_size=5, ax=ax)

        # Etiquetas de aristas también opcionalmente limitadas

        edge_labels = nx.get_edge_attributes(self.graph, 'weight')
        if len(self.graph.edges) <= 1000:
            nx.draw_networkx_edge_labels(self.graph, self.pos, edge_labels={(u, v): f"{d:.1f}" for (u, v), d in edge_labels.items()},
                                         font_size=3, ax=ax)

        # plt.colorbar(nodes, ax=ax, label='Grado del nodo')
        ax.set_title("Grafo de distancias entre nodos (heatmap por grado)", fontsize=16)
        plt.axis('off')
        plt.tight_layout()

        if guardar_figura:
            plt.show()
            fig.savefig(nombre_figura, dpi=300)
            print(f"✅ Figura guardada como: {nombre_figura}")
        else:
            plt.show()

