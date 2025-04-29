from model.NodeClass import Node
from model.LinkClass import Link
from model.NetworkGraph import CompleteGraph

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

import geopandas as gpd
import contextily as ctx
from shapely.geometry import Point, LineString
import numpy as np


class DefaultFigureGenerator:
    pass
    
class SimpleNetworkPlot:
    pass



# -------------------------------------------------------------------------------------- #
# Node Degree Heat Map Plot                                                              #
# -------------------------------------------------------------------------------------- #
class NodeDegreeHeatMapPlot:
    
    @staticmethod
    def plot_nodes(
        nodes: list[Node], 
        fig: plt.Figure, 
        ax: plt.Axes, 
        include_node_labels: bool = True,
        hl4_shape: str = '*',
        hl4_size: int = 150,
        hl4_color: str = 'red',
        hl5_shape: str = 'o',
        hl5_size: int = 100,
        include_color_bar: bool = True,
    ):
        
        x, y = Node.obtain_x_y_vectors(nodes)
        degrees = Node.obtain_degree_vector(nodes)
        types = Node.obtain_type_vector(nodes)
        names = Node.obtain_name_vector(nodes)

        hl4_indices = [i for i, t in enumerate(types) if t == "HL4"]
        hl5_indices = [i for i, t in enumerate(types) if t == "HL5"]
        hl5_degrees = [degrees[i] for i in hl5_indices]
        
        # Normalize degrees for color mapping
        norm = plt.Normalize(min(hl5_degrees), max(hl5_degrees))
        cmap = plt.cm.viridis
        hl5_colors = [cmap(norm(deg)) for deg in hl5_degrees]
        
        # Plot hl4 nodes
        ax.scatter([x[i] for i in hl4_indices],
                [y[i] for i in hl4_indices],
                c=hl4_color,
                s=hl4_size,
                marker=hl4_shape,
                label='HL4')
        
        # Plot hl5 nodes
        ax.scatter([x[i] for i in hl5_indices],
                [y[i] for i in hl5_indices],
                c=hl5_colors,
                s=hl5_size,
                marker=hl5_shape,
                label='HL5')
        
        if include_node_labels:
            for i, name in enumerate(names):
                ax.text(x[i], y[i], name, fontsize=6, ha='right', va='bottom')
                
        if include_color_bar:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            plt.colorbar(sm, ax=ax, label='Node Degree (HL5 nodes only)')
            
    @staticmethod
    def plot_links(
        links: list[Link], 
        fig: plt.Figure, 
        ax: plt.Axes, 
        include_edge_labels: bool = True,
        edge_color: str = 'gray',
        edge_width: int = 1,
        edge_style: str = 'k-',
        text_color: str = 'gray',
        text_size: int = 6,
    ):
        for link in links:
            a = link.a.pos
            b = link.b.pos
            
            ax.plot([a[0], b[0]], [a[1], b[1]], edge_style, lw=edge_width, color=edge_color)
            if include_edge_labels:
                ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, link.label, fontsize=text_size, color=text_color)
    
    @staticmethod
    def plot(
            graph: CompleteGraph,
            fig: plt.Figure = None,
            ax: plt.Axes = None,
            include_node_labels: bool = False,
            include_edge_labels: bool = True,
            hl4_shape: str = '*',
            hl4_size: int = 150,
            hl4_color: str = 'red',
            hl5_shape: str = 'o',
            hl5_size: int = 100,
        ):
        is_own_figure = fig is None or ax is None
        if is_own_figure:
            fig, ax = plt.subplots(figsize=(40, 40))
            
        # Plot nodes
        NodeDegreeHeatMapPlot.plot_nodes(
            nodes=graph.nodes,
            fig=fig,
            ax=ax,
            include_node_labels=include_node_labels,
            hl4_shape=hl4_shape,
            hl4_size=hl4_size,
            hl4_color=hl4_color,
            hl5_shape=hl5_shape,
            hl5_size=hl5_size,
            include_color_bar=is_own_figure,
        )

        # Plot links
        NodeDegreeHeatMapPlot.plot_links(
            links=graph.links,
            fig=fig,
            ax=ax,
            include_edge_labels=include_edge_labels,
        )
        
        
                
        if is_own_figure:
            # Create custom legend handles
            hl5_patch = mpatches.Patch(color='blue', label='HL5 (colored by degree)')
            hl4_patch = mlines.Line2D([], [], color='red', marker=hl4_shape, linestyle='None', markersize=10, label='HL4')
        
            # Add the legend to the plot
            fig.legend(handles=[hl5_patch, hl4_patch], loc='upper right')        
        
        if is_own_figure:
            ax.set_title("Grafo de Red", fontsize=16)
            fig.tight_layout()
            fig.show()
        
        return fig, ax
    



# -------------------------------------------------------------------------------------- #
# Real Map Network Plot                                                                  #
# -------------------------------------------------------------------------------------- #
class RealMapNetworkPlot:
    
    
    def plot(
        graph: CompleteGraph,
        fig: plt.Figure = None,
        ax: plt.Axes = None,
        guardar_figura: bool = True,
        nombre_figura: str = "red_real_map.png",
        include_node_labels: bool = False,
        include_edge_labels: bool = True,
        scale_factor: float = 10,
        lat_center: float = 40.4168,    # Madrid Puerta del Sol (40.4168, -3.7038)
        lon_center: float = -3.7038,    # Madrid Puerta del Sol (40.4168, -3.7038)
    ):
        # Is own figure
        is_own_figure = fig is None or ax is None
        if is_own_figure:
            fig, ax = plt.subplots(figsize=(40, 40))
            
        graph = graph.model_copy()
        
        # Convert km to geographic coordinates
        deg_per_km_lat = 1 / 111
        deg_per_km_lon = 1 / (111 * np.cos(np.radians(lat_center)))
        
        # Create GeoDataFrame for nodes
        node_geometries = []
        for node in graph.nodes:
            node.pos = (
                lon_center + (node.pos[0] * scale_factor) * deg_per_km_lon,
                lat_center + (node.pos[1] * scale_factor) * deg_per_km_lat
            )
            node_geometries.append(Point(node.pos[0], node.pos[1]))
        
        # Create GeoDataFrame for nodes
        gdf_nodes = gpd.GeoDataFrame(
            
            geometry=node_geometries,
            crs="EPSG:4326"
        ).to_crs(epsg=3857)
        
        # Create GeoDataFrame for links
        link_geometries = []
        for link in graph.links:
            a = link.a.pos
            b = link.b.pos
            link_geometries.append(LineString([(a[0], a[1]), (b[0], b[1])]))
        
        gdf_links = gpd.GeoDataFrame(
            geometry=link_geometries,
            crs="EPSG:4326"
        ).to_crs(epsg=3857)
        
        # Plot links first
        gdf_links.plot(ax=ax, color='gray', linewidth=1, alpha=0.5)
        
        # Plot nodes
        NodeDegreeHeatMapPlot.plot_nodes(
            nodes=graph.nodes,
            fig=fig,
            ax=ax,
            include_node_labels=include_node_labels,
            include_color_bar=is_own_figure,
        )
        
        # Add basemap
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom=12)
        
        # Set proper limits based on the GeoDataFrame bounds
        bounds = gdf_nodes.total_bounds
        margin = 0.1  # 10% margin
        x_margin = (bounds[2] - bounds[0]) * margin
        y_margin = (bounds[3] - bounds[1]) * margin
        
        ax.set_xlim(bounds[0] - x_margin, bounds[2] + x_margin)
        ax.set_ylim(bounds[1] - y_margin, bounds[3] + y_margin)
        
        # Set title and show
        ax.set_title("Red de nodos sobre mapa real", fontsize=15)
        
        if is_own_figure:
            fig.show()
            if guardar_figura:
                fig.savefig(nombre_figura, dpi=300)
                        
                
        
         
    
    