# --- DEPENDENCIAS ---
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


# --- FUNCIÓN PRINCIPAL ---
def crear_figura(G, pos, node_colors, edge_labels, guardar_figura, nombre_figura):
    fig, ax = plt.subplots(figsize=(40, 40))
    nodes = nx.draw_networkx_nodes(G, pos, node_size=600, node_color=node_colors, cmap=plt.cm.viridis, ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax)

    # Mostrar etiquetas de nodos si hay pocos
    if len(G.nodes) <= 200:
        nx.draw_networkx_labels(G, pos, {i: f"N{i + 1}" for i in G.nodes()}, font_size=5, ax=ax)

    # Etiquetas de aristas también opcionalmente limitadas
    if len(G.edges) <= 1000:
        nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): f"{d:.1f}" for (u, v), d in edge_labels.items()},
                                     font_size=3, ax=ax)

    plt.colorbar(nodes, ax=ax, label='Grado del nodo')
    ax.set_title("Grafo de distancias entre nodos (heatmap por grado)", fontsize=16)
    plt.axis('off')
    plt.tight_layout()

    if guardar_figura:
        plt.show()
        fig.savefig(nombre_figura, dpi=300)
        print(f"✅ Figura guardada como: {nombre_figura}")
    else:
        plt.show()


def visualizar_matriz_distancias(path_mat, guardar_figura=True, nombre_figura="grafo_distancias.png"):
    """
    Visualiza una matriz de distancias entre nodos desde un archivo .mat como un grafo.
    Los nodos se colorean según su grado (número de conexiones).

    Parámetros:
    - path_mat: str, ruta al archivo .mat que contiene 'crossMatrix'
    - guardar_figura: bool, si True guarda la imagen en disco
    - nombre_figura: str, nombre del archivo de salida si guardar_figura es True
    """
    # Cargar archivo .mat
    mat_contents = scipy.io.loadmat(path_mat)
    distance_matrix = mat_contents['crossMatrix']

    # Crear el grafo
    G = nx.Graph()
    for i in range(distance_matrix.shape[0]):
        G.add_node(i, label=f"N{i + 1}")
    for i in range(distance_matrix.shape[0]):
        for j in range(i + 1, distance_matrix.shape[1]):
            distance = distance_matrix[i, j]
            if distance > 0:
                G.add_edge(i, j, weight=distance)

    # Layout y métricas
    pos = nx.spring_layout(G, seed=42)
    grados = dict(G.degree())
    node_colors = [grados[n] for n in G.nodes()]
    edge_labels = nx.get_edge_attributes(G, 'weight')

    # Llamar a la nueva función para crear la figura
    crear_figura(G, pos, node_colors, edge_labels, guardar_figura, nombre_figura)


if __name__ == "__main__":
    visualizar_matriz_distancias("Passion_Xtreme_III.mat", False)
