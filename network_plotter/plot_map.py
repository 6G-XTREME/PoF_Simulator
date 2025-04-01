import scipy.io
import pandas as pd
import json
import networkx as nx
from model.NodeClass import Node
from model.FileFormat import FileFormat
from model.LinkClass import Link
import matplotlib.pyplot as plt


def build_graph(distance_matrix, xlsx_data):
    G = nx.Graph()
    for i in range(distance_matrix.shape[0]):
        G.add_node(i)
    links = []
    for i in range(distance_matrix.shape[0]):
        for j in range(i + 1, distance_matrix.shape[1]):
            distance = distance_matrix[i, j]
            if distance > 0:
                G.add_edge(i, j, weight=distance)
                links.append(Link(source=i, target=j, distance=distance, color='blue', label=f'{distance:.2f}km'))
                
    nodes = []
    pos = nx.spring_layout(G, seed=42)
    for i in range(distance_matrix.shape[0]):
        node_id = i
        node_name = xlsx_data.iloc[i, 0]
        node_type = xlsx_data.iloc[i, 1]
        node_degree = int(sum(distance_matrix[i] > 0))
        node_x, node_y = pos[i]
        nodes.append(Node(id=node_id, type=node_type, x=node_x, y=node_y, node_degree=node_degree, color='blue', size=10, shape='circle', label=node_name))
        
    return G, links, nodes

def merge_and_create_nodes(mat_path, xlsx_path, json_output_path):
    # Load the .mat file
    mat_contents = scipy.io.loadmat(mat_path)
    distance_matrix = mat_contents['crossMatrix']

    # Load the .xlsx file
    xlsx_data = pd.read_excel(xlsx_path)

    # Create the graph
    G = nx.Graph()
    for i in range(distance_matrix.shape[0]):
        G.add_node(i)
    links = []
    for i in range(distance_matrix.shape[0]):
        for j in range(i + 1, distance_matrix.shape[1]):
            distance = distance_matrix[i, j]
            if distance > 0:
                G.add_edge(i, j, weight=distance)
                # Create a Link instance
                link = Link(source=i, target=j, distance=distance, color='blue', label=f'{distance:.2f}km')
                links.append(link)

    # Calculate node positions using spring layout
    pos = nx.spring_layout(G, seed=42)

    # Create a list of Node objects
    nodes = []
    for i in range(1, distance_matrix.shape[0]):  # Skip the first row
        node_id = i
        node_name = xlsx_data.iloc[i, 0]
        node_type = xlsx_data.iloc[i, 1]
        node_degree = int(sum(distance_matrix[i] > 0))
        node_x, node_y = pos[i]

        # Create a Node instance
        node = Node(
            id=node_id,
            type=node_type,
            x=node_x,
            y=node_y,
            node_degree=node_degree,
            color='blue',  # Placeholder for color
            size=10,  # Placeholder for size
            shape='circle',  # Placeholder for shape
            label=node_name
        )
        nodes.append(node)

    # Create a ModelStorage instance
    model_storage = FileFormat(nodes=nodes, links=links)

    # Convert the ModelStorage object to JSON
    model_storage_json = model_storage.dict()
    with open(json_output_path, 'w') as json_file:
        json.dump(model_storage_json, json_file, indent=4)

    print(f"✅ Merged information saved to: {json_output_path}")
    
    
    nx.draw(G, pos, with_labels=True)
    plt.show()

if __name__ == "__main__":
    merge_and_create_nodes("Passion_Xtreme_III.mat", "NameTypes.xlsx", "merged_nodes.json")