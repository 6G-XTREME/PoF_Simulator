from model.LinkClass import Link
from model.NodeClass import Node
from model.FileFormat import FileFormat
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io

class GraphComplete:
    graph_functions = {
        "spring_layout": lambda G: nx.spring_layout(G, seed=42),
        "circular_layout": lambda G: nx.circular_layout(G),             # TODO: tune
        "kamada_kawai_layout": lambda G: nx.kamada_kawai_layout(G),     # TODO: tune
        "random_layout": lambda G: nx.random_layout(G),                 # TODO: tune
        "shell_layout": lambda G: nx.shell_layout(G),                   # TODO: tune
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
        
        self.discarded_nodes = []
        
        for i in range(distance_matrix.shape[0]):
            if xlsx_data.iloc[i, 1] != "HL4" and xlsx_data.iloc[i, 1] != "HL5":
                self.discarded_nodes.append(i)

        print(f"Discarded nodes: {self.discarded_nodes}")
        
        for i in range(distance_matrix.shape[0]):
            if i not in self.discarded_nodes:
                self.graph.add_node(i)
        
        for i in range(distance_matrix.shape[0]):
            for j in range(i + 1, distance_matrix.shape[1]):
                if i not in self.discarded_nodes and j not in self.discarded_nodes:
                    distance = distance_matrix[i, j]
                    if distance > 0:
                        self.graph.add_edge(i, j, weight=1/distance)
                    self.links.append(Link(source=i, target=j, distance=distance, label=f'{distance:.2f}km'))
                    
        self.layout_function = layout_function
        self.pos = self.graph_functions[layout_function](self.graph)
        
        
        for i in range(distance_matrix.shape[0]):
            if i not in self.discarded_nodes:
                node_id = i
                node_name = xlsx_data.iloc[i, 0]
                node_type = xlsx_data.iloc[i, 1]
                node_degree = int(sum(distance_matrix[i] > 0))
                node_x, node_y = self.pos[i]
            self.nodes.append(Node(id=node_id, type=node_type, x=node_x, y=node_y, node_degree=node_degree, name=node_name))
        

    def to_json(self):
        return FileFormat(nodes=self.nodes, links=self.links).model_dump()
    
    
    def plot_graph(self, guardar_figura=True, nombre_figura="grafo_distancias.png"):
        fig, ax = plt.subplots(figsize=(40, 40))
        node_colors = ["yellow" if node.type == "HL4" else "green" if node.type == "HL5" else "blue" for node in self.nodes]
        node_colors = [node_colors[i] for i in range(len(node_colors)) if i not in self.discarded_nodes]
        nodes = nx.draw_networkx_nodes(self.graph, self.pos, node_size=100, node_color=node_colors, cmap=plt.cm.viridis, ax=ax)
        nx.draw_networkx_edges(self.graph, self.pos, alpha=0.3, ax=ax)

        # Mostrar etiquetas de nodos si hay pocos
        if len(self.graph.nodes) <= 200:
            nx.draw_networkx_labels(self.graph, self.pos, {i: f"{self.nodes[i].name}" for i in self.graph.nodes()}, font_size=5, ax=ax)

        # Etiquetas de aristas también opcionalmente limitadas
        edge_labels = nx.get_edge_attributes(self.graph, 'weight')
        edge_labels = {(u, v): f"{1/d:.1f} km" for (u, v), d in edge_labels.items()}
        if len(self.graph.edges) <= 1000:
            nx.draw_networkx_edge_labels(self.graph, self.pos, edge_labels=edge_labels, font_size=3, ax=ax)

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

