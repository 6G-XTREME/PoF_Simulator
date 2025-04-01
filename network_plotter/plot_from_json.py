import json
import matplotlib.pyplot as plt
import networkx as nx
from models.NodeClass import Node, Link, ModelStorage

def plot_from_json(input_data, is_file_path=False):
    """
    Plots a graph from a JSON representation of nodes and links.

    Parameters:
    - input_data: Can be a dict (JSON), a JSON file path, or a ModelStorage object.
    - is_file_path: Boolean indicating if input_data is a file path.
    """
    # Load data based on input type
    if is_file_path:
        with open(input_data, 'r') as file:
            model_data = json.load(file)
    elif isinstance(input_data, dict):
        model_data = input_data
    elif isinstance(input_data, ModelStorage):
        model_data = input_data.dict()
    else:
        raise ValueError("Invalid input data type.")

    # Extract nodes and links
    nodes_data = model_data['nodes']
    links_data = model_data['links']

    # Create the graph
    G = nx.Graph()
    for node_data in nodes_data:
        node_id = node_data['id']
        G.add_node(node_id, label=node_data['label'], type=node_data['type'])

    for link_data in links_data:
        source = link_data['source']
        target = link_data['target']
        weight = link_data['weight']
        G.add_edge(source, target, weight=weight)

    # Calculate node positions using spring layout
    pos = {node_data['id']: (node_data['x'], node_data['y']) for node_data in nodes_data}

    # Plot the graph
    fig, ax = plt.subplots(figsize=(10, 10))
    node_colors = [node_data['node_degree'] for node_data in nodes_data]
    nx.draw_networkx_nodes(G, pos, node_size=100, node_color=node_colors, cmap=plt.cm.viridis, ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax)
    nx.draw_networkx_labels(G, pos, {node_data['id']: node_data['label'] for node_data in nodes_data}, font_size=8, ax=ax)

    plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.viridis), ax=ax, label='Node Degree')
    ax.set_title("Graph from JSON Nodes and Links", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Load from JSON file
    plot_from_json("merged_nodes.json", is_file_path=True)

    # Load from JSON dict
    with open("merged_nodes.json", 'r') as file:
        json_data = json.load(file)
    plot_from_json(json_data)

    # Load from ModelStorage object
    model_storage = ModelStorage(**json_data)
    plot_from_json(model_storage) 