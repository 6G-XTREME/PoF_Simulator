from model.LinkClass import Link
from model.NodeClass import Node
from model.FileFormat import FileFormat
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io
import geopandas as gpd
import contextily as ctx
from shapely.geometry import LineString, Point


class GraphComplete:
    graph_functions = {
        "spring_layout": lambda G: nx.spring_layout(G, seed=42),
        "circular_layout": lambda G: nx.circular_layout(G),             # TODO: tune
        "kamada_kawai_layout": lambda G: nx.kamada_kawai_layout(G),     # TODO: tune
        "random_layout": lambda G: nx.random_layout(G),                 # TODO: tune
        "shell_layout": lambda G: nx.shell_layout(G),                   # TODO: tune
    }
    

    # ---------------------------------------------------------------------------------------------------------------- #
    # -- Wrapper to create from files -------------------------------------------------------------------------------- #
    #                                                                                                                  #
    # ---------------------------------------------------------------------------------------------------------------- #
    @staticmethod
    def of(distance_matrix_path: str, xlsx_data_path: str, layout_function: str = "spring_layout"):
        distance_matrix = scipy.io.loadmat(distance_matrix_path)['crossMatrix']
        xlsx_data = pd.read_excel(xlsx_data_path)
        return GraphComplete(distance_matrix, xlsx_data, layout_function)
    
    
    # ---------------------------------------------------------------------------------------------------------------- #
    # -- Constructor ------------------------------------------------------------------------------------------------- #
    #                                                                                                                  #
    # ---------------------------------------------------------------------------------------------------------------- #
    def __init__(self, distance_matrix: np.ndarray, xlsx_data: pd.DataFrame, layout_function: str = "spring_layout"):
        self.graph = nx.Graph()
        self.links = []
        self.nodes = []
        self.nodes_to_discard = []
        self.links_to_discard = []
        
        # Coordinates for Puerta del Sol
        puerta_del_sol_lat = 40.4168
        puerta_del_sol_lon = -3.7038

        for i in range(distance_matrix.shape[0]):
            if xlsx_data.iloc[i, 1] != "HL4" and xlsx_data.iloc[i, 1] != "HL5":
                self.nodes_to_discard.append(i)

        for i in range(distance_matrix.shape[0]):
            if i not in self.nodes_to_discard:
                self.graph.add_node(i)
        
        for i in range(distance_matrix.shape[0]):
            for j in range(i + 1, distance_matrix.shape[1]):
                distance = distance_matrix[i, j]
                if distance > 0:
                    if i not in self.nodes_to_discard and j not in self.nodes_to_discard:
                        self.graph.add_edge(i, j, weight=1/distance)
                    else:
                        self.links_to_discard.append(len(self.links))
                    
                    self.links.append(Link(source=i, target=j, distance=distance, label=f'{distance:.2f}km'))

        # Compute layout positions
        self.layout_function = layout_function
        self.pos = self.graph_functions[layout_function](self.graph)

        for i in range(distance_matrix.shape[0]):
            if i not in self.nodes_to_discard:
                node_id = i
                node_name = xlsx_data.iloc[i, 0]
                node_type = xlsx_data.iloc[i, 1]
                node_degree = int(sum(distance_matrix[i] > 0))
                node_x, node_y = self.pos[i]

                self.nodes.append(Node(id=node_id, type=node_type, x=node_x, y=node_y, node_degree=node_degree, name=node_name))

        print(f'Nodes: {len(self.nodes)}')
        print(f'Links: {len(self.links)}')
        print(f'Discarded nodes: {len(self.nodes_to_discard)}')
        print(f'Discarded links: {len(self.links_to_discard)}')
        print(f'Remaining nodes: {len(self.graph.nodes)}')
        print(f'Remaining links: {len(self.graph.edges)}')
        

    # ---------------------------------------------------------------------------------------------------------------- #
    # -- To JSON ----------------------------------------------------------------------------------------------------- #
    #                                                                                                                  #
    # ---------------------------------------------------------------------------------------------------------------- #
    def to_json(self):
        return FileFormat(nodes=self.nodes, links=self.links).model_dump()
    
    
    
    # ---------------------------------------------------------------------------------------------------------------- #
    # -- Plot without map -------------------------------------------------------------------------------------------- #
    #                                                                                                                  #
    # ---------------------------------------------------------------------------------------------------------------- #
    def plot_graph_on_map(self, guardar_figura=True, nombre_figura="grafo_distancias.png"):
        fig, ax = plt.subplots(figsize=(40, 40))
        node_colors = ["yellow" if node.type == "HL4" else "green" if node.type == "HL5" else "blue" for node in self.nodes]
        node_colors = [node_colors[i] for i in range(len(node_colors)) if i not in self.nodes_to_discard]
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
            
    
    
    # ---------------------------------------------------------------------------------------------------------------- #
    # -- Plot with map ----------------------------------------------------------------------------------------------- #
    #                                                                                                                  #
    # ---------------------------------------------------------------------------------------------------------------- #
    def plot_graph_with_map(self, guardar_figura=True, nombre_figura="grafo_distancias.png"):
        # 1. Centro del grafo en 0,0
        pos_array = np.array(list(self.pos.values()))
        pos_centered = pos_array - pos_array.mean(axis=0)
    
        # 2. Escalado: ajusta este valor si los nodos están muy separados o muy juntos
        scale_factor = 10  # en km
        pos_km = pos_centered * scale_factor
    
        # 3. Conversión de km a grados geográficos
        lat_center = 40.4168
        lon_center = -3.7038
        deg_per_km_lat = 1 / 111
        deg_per_km_lon = 1 / (111 * np.cos(np.radians(lat_center)))
    
        pos_latlon = {
            node_id: (
                lon_center + x * deg_per_km_lon,
                lat_center + y * deg_per_km_lat
            )
            for (node_id, (x, y)) in zip(self.pos.keys(), pos_km)
        }
    
        # 4. GeoDataFrame de nodos
        node_colors = ["yellow" if node.type == "HL4" else "green" if node.type == "HL5" else "blue" for node in self.nodes]
    
        gdf_nodes = gpd.GeoDataFrame(
            {
                "id": [n.id for n in self.nodes],
                "name": [n.name for n in self.nodes],
                "color": node_colors,
            },
            geometry=[Point(pos_latlon[n.id][0], pos_latlon[n.id][1]) for n in self.nodes],
            crs="EPSG:4326"
        ).to_crs(epsg=3857)
    
        # 5. GeoDataFrame de aristas
        lines = []
        edge_labels = []
    
        for link in self.links:
            if link.source in pos_latlon and link.target in pos_latlon:
                source_coords = pos_latlon[link.source]
                target_coords = pos_latlon[link.target]
                line = LineString([source_coords, target_coords])
                lines.append(line)
                edge_labels.append(f"{link.distance:.1f} km")
    
        gdf_edges = gpd.GeoDataFrame(
            {"label": edge_labels},
            geometry=lines,
            crs="EPSG:4326"
        ).to_crs(epsg=3857)
    
        # 6. Plot
        fig, ax = plt.subplots(figsize=(15, 15))
        fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

    
        # Dibujar edges primero
        gdf_edges.plot(ax=ax, linewidth=0.8, alpha=0.5, color='gray')
    
        # Dibujar nodos
        gdf_nodes.plot(ax=ax, color=gdf_nodes["color"], markersize=30)
    
        # Mostrar nombres de nodos si hay pocos
        # if len(self.nodes) <= 200:
            # for x, y, name in zip(gdf_nodes.geometry.x, gdf_nodes.geometry.y, gdf_nodes["name"]):
                # ax.text(x, y, name, fontsize=6, ha='right', va='bottom')
    
        # Mostrar etiquetas de aristas si hay pocas
        if len(gdf_edges) <= 1000:
            for geom, label in zip(gdf_edges.geometry, gdf_edges["label"]):
                x, y = geom.interpolate(0.5, normalized=True).xy
                ax.text(x[0], y[0], label, fontsize=5, color='gray')
    
        # Añadir mapa base
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
    
        # Zoom automático a los nodos# Zoom automático a los nodos (con margen)
        margin_x = (gdf_nodes.total_bounds[2] - gdf_nodes.total_bounds[0]) * 0.1  # 10% extra
        margin_y = (gdf_nodes.total_bounds[3] - gdf_nodes.total_bounds[1]) * 0.1
        ax.set_xlim(gdf_nodes.total_bounds[0] - margin_x, gdf_nodes.total_bounds[2] + margin_x)
        ax.set_ylim(gdf_nodes.total_bounds[1] - margin_y, gdf_nodes.total_bounds[3] + margin_y)

    
        ax.set_title("Red de nodos sobre mapa real", fontsize=15)
        ax.axis("off")
    
        plt.show()
        if guardar_figura:
            fig.savefig(nombre_figura, dpi=300)
            print(f"✅ Mapa guardado como: {nombre_figura}")
            


