from shapely.geometry import MultiPolygon, Polygon, GeometryCollection
import matplotlib.pyplot as plt
from model.NodeClass import Node
import numpy as np
from numpy.typing import NDArray
from matplotlib.lines import Line2D


def standard_plot(
        regions:list,
        painting_nodes:list[Node],
        scaling_factor:int = 1,
        canvas: tuple[plt.Figure, plt.Axes] = None,
        coverage_area_config: dict = {},
        macrocell_config: dict = {},
        femtocell_config: dict = {},
        plot_config: dict = {},
        extra_plot_functions: list[callable] = [],
    ):
    is_own_figure = canvas is None
    if is_own_figure:
        fig, ax = plt.subplots(figsize=plot_config.get("figsize", (10, 10)))
    else:
        fig, ax = canvas
    
    final_cover_area_config = {
        "alpha": coverage_area_config.get("alpha", 0.3),
        "edgecolor": coverage_area_config.get("edgecolor", 'black'),
        "linewidth": coverage_area_config.get("linewidth", 0.5),
        "linestyle": coverage_area_config.get("linestyle", '--'),
    }
    final_macrocell_config = {
        "marker": macrocell_config.get("marker", "*"),
        "s": macrocell_config.get("s", 50),
        "color": macrocell_config.get("color", 'red'),
        "alpha": macrocell_config.get("alpha", 1),
    }
    final_femtocell_config = {
        "marker": femtocell_config.get("marker", "o"),
        "s": femtocell_config.get("s", 30),
        "color": femtocell_config.get("color", 'blue'),
        "alpha": femtocell_config.get("alpha", 1),
    }
    def config_from_type(node):
        return final_macrocell_config if node.type == "HL4" else final_femtocell_config
    
    
    # Paint the regions (bs areas)
    for i in range(len(regions)-1, -1, -1):
        region = regions[i]
        if isinstance(region, Polygon):
            x, y = region.exterior.coords.xy
            ax.fill(x, y, **final_cover_area_config)
        elif isinstance(region, MultiPolygon):
            for reg in region.geoms:
                x, y = reg.exterior.coords.xy
                ax.fill(x, y, **final_cover_area_config)
        elif isinstance(region, GeometryCollection):
            for geom in region.geoms:
                if isinstance(geom, Polygon):
                    x, y = geom.exterior.coords.xy
                    ax.fill(x, y, **final_cover_area_config)
        else:
            x, y = region.exterior.coords.xy
            ax.fill(x, y, **final_cover_area_config)
    
    
    # Paint the nodes (bs)
    for node in painting_nodes:
        ax.scatter(node.pos[0] * scaling_factor, node.pos[1] * scaling_factor, **config_from_type(node))
        
        
    # Extra plot settings, is the figure is own
    if is_own_figure:
        
        # Adjust axis ticks to show real scale values
        # current_xticks = ax.get_xticks()
        # current_yticks = ax.get_yticks()
        # ax.xaxis.set_major_locator(plt.FixedLocator(current_xticks))
        # ax.yaxis.set_major_locator(plt.FixedLocator(current_yticks))
        # ax.xaxis.set_major_formatter(plt.FixedFormatter([f'{round(x/scaling_factor, 1)}' for x in current_xticks]))
        # ax.yaxis.set_major_formatter(plt.FixedFormatter([f'{round(y/scaling_factor, 1)}' for y in current_yticks]))
        ax.set_xlabel(f"x ({scaling_factor} km)")
        ax.set_ylabel(f"y ({scaling_factor} km)")
        
        # Add title
        fig.suptitle(plot_config.get("title", "Regions for base stations"))
        
        # Add legend for cell types
        ax.scatter([], [], label="Macrocell", **final_macrocell_config)
        ax.scatter([], [], label="Femtocell", **final_femtocell_config)
        ax.legend()
        
        for extra_plot_function in extra_plot_functions:
            extra_plot_function(fig, ax)
        
        plt.show()
    
    return fig, ax
    
    
    
    

def plot_algorithm_result(
        regions:list,
        nodes: list[tuple[float, float, bool, bool, bool]], # (x, y, has_macrocell, has_femtocell, has_hpld)
        canvas: tuple[plt.Figure, plt.Axes] = None,
        coverage_area_config: dict = {},
        macrocell_config: dict = {},
        femtocell_config: dict = {},
        hpld_config: dict = {},
        plot_config: dict = {},
        extra_plot_functions: list[callable] = [],
        scaling_factor:int = 1,
    ):
    is_own_figure = canvas is None
    if is_own_figure:
        fig, ax = plt.subplots(figsize=plot_config.get("figsize", (10, 10)))
    else:
        fig, ax = canvas
    
    final_cover_area_config = {
        "alpha": coverage_area_config.get("alpha", 0.3),
        "edgecolor": coverage_area_config.get("edgecolor", 'black'),
        "linewidth": coverage_area_config.get("linewidth", 0.5),
        "linestyle": coverage_area_config.get("linestyle", '--'),
    }
    
    
    
    # Paint the regions (bs areas)
    for i in range(len(regions)-1, -1, -1):
        region = regions[i]
        if isinstance(region, Polygon):
            x, y = region.exterior.coords.xy
            ax.fill(x, y, **final_cover_area_config)
        elif isinstance(region, MultiPolygon):
            for reg in region.geoms:
                x, y = reg.exterior.coords.xy
                ax.fill(x, y, **final_cover_area_config)
        elif isinstance(region, GeometryCollection):
            for geom in region.geoms:
                if isinstance(geom, Polygon):
                    x, y = geom.exterior.coords.xy
                    ax.fill(x, y, **final_cover_area_config)
        else:
            x, y = region.exterior.coords.xy
            ax.fill(x, y, **final_cover_area_config)
    
    

    # node = (x, y, has_macrocell, has_femtocell, has_hpld)
    #         0  1  2              3              4
    for node in nodes:
        # Star if has macrocell, diamond if has femtocell, circle if hosts no cell
        marker = "*" if node[2] else "d" if node[3] else "o"
        color = "red" if node[4] else "black"
        ax.scatter(node[0] * scaling_factor, node[1] * scaling_factor, marker=marker, s=50, color=color)
        
    
        
        
    # Extra plot settings, is the figure is own
    if is_own_figure:
        
        # Adjust axis ticks to show real scale values
        # current_xticks = ax.get_xticks()
        # current_yticks = ax.get_yticks()
        # ax.xaxis.set_major_locator(plt.FixedLocator(current_xticks))
        # ax.yaxis.set_major_locator(plt.FixedLocator(current_yticks))
        # ax.xaxis.set_major_formatter(plt.FixedFormatter([f'{round(x/scaling_factor, 1)}' for x in current_xticks]))
        # ax.yaxis.set_major_formatter(plt.FixedFormatter([f'{round(y/scaling_factor, 1)}' for y in current_yticks]))
        ax.set_xlabel(f"x ({scaling_factor} km)")
        ax.set_ylabel(f"y ({scaling_factor} km)")
        
        # Add title
        fig.suptitle(plot_config.get("title", "Regions for base stations"))
        
        # Add legend for cell types
        ax.scatter([], [], label="Has Macrocell", marker="*", s=50, color="red")
        ax.scatter([], [], label="Has Femtocell", marker="d", s=30, color="blue")
        ax.scatter([], [], label="Has HPLD (size 50)", marker="o", s=50, color="green")
        ax.scatter([], [], label="Has no HPLD (size 30)", marker="o", s=30, color="black")
        ax.legend()
        
        for extra_plot_function in extra_plot_functions:
            extra_plot_function(fig, ax)
        
        plt.show()
    
    return fig, ax
    
    

def plot_algorithm_result_associations(
        regions:list,
        nodes: list[tuple[float, float, bool, bool, bool]], # (x, y, has_macrocell, has_femtocell, has_hpld)
        associations: NDArray[np.int_],
        canvas: tuple[plt.Figure, plt.Axes] = None,
        coverage_area_config: dict = {},
        macrocell_config: dict = {},
        femtocell_config: dict = {},
        hpld_config: dict = {},
        plot_config: dict = {},
        association_config: dict = {"alpha": 0.5, "c": 'black', "lw": 0.5},
        extra_plot_functions: list[callable] = [],
        scaling_factor:int = 1,
    ):
    is_own_figure = canvas is None
    if is_own_figure:
        fig, ax = plt.subplots(figsize=plot_config.get("figsize", (10, 10)))
    else:
        fig, ax = canvas
    
    final_cover_area_config = {
        "alpha": coverage_area_config.get("alpha", 0.3),
        "edgecolor": coverage_area_config.get("edgecolor", 'black'),
        "linewidth": coverage_area_config.get("linewidth", 0.5),
        "linestyle": coverage_area_config.get("linestyle", '--'),
    }
    
    
    
    # Paint the regions (bs areas)
    for i in range(len(regions)-1, -1, -1):
        region = regions[i]
        if isinstance(region, Polygon):
            x, y = region.exterior.coords.xy
            ax.fill(x, y, **final_cover_area_config)
        elif isinstance(region, MultiPolygon):
            for reg in region.geoms:
                x, y = reg.exterior.coords.xy
                ax.fill(x, y, **final_cover_area_config)
        elif isinstance(region, GeometryCollection):
            for geom in region.geoms:
                if isinstance(geom, Polygon):
                    x, y = geom.exterior.coords.xy
                    ax.fill(x, y, **final_cover_area_config)
        else:
            x, y = region.exterior.coords.xy
            ax.fill(x, y, **final_cover_area_config)
    
    

    # node = (x, y, has_macrocell, has_femtocell, has_hpld)
    #         0  1  2              3              4
    for node in nodes:
        # Star if has macrocell, diamond if has femtocell, circle if hosts no cell
        marker = "*" if node[2] else "d" if node[3] else "o"
        color = "red" if node[4] else "black"
        ax.scatter(node[0] * scaling_factor, node[1] * scaling_factor, marker=marker, s=50, color=color)
        
    # Plot the associations
    for i in range(associations.shape[0]):
        for j in range(associations.shape[1]):
            if associations[i, j] == 1:
                ax.plot([nodes[i][0] * scaling_factor, nodes[j][0] * scaling_factor], [nodes[i][1] * scaling_factor, nodes[j][1] * scaling_factor], **association_config)
        
    
        
        
    # Extra plot settings, is the figure is own
    if is_own_figure:
        
        ax.set_xlabel(f"x ({scaling_factor} km)")
        ax.set_ylabel(f"y ({scaling_factor} km)")
        
        # Add title
        fig.suptitle(plot_config.get("title", "Regions for base stations"))
        
        # Add legend for cell types
        # ax.scatter([], [], label="Has Macrocell", marker="*", s=50, color="red")
        # ax.scatter([], [], label="Has Femtocell", marker="d", s=30, color="blue")
        # ax.scatter([], [], label="Has HPLD (size 50)", marker="o", s=50, color="green")
        # ax.scatter([], [], label="Has no HPLD (size 30)", marker="o", s=30, color="black")
        # ax.legend()
        
        for extra_plot_function in extra_plot_functions:
            extra_plot_function(fig, ax)
        
        plt.show()
    
    return fig, ax

    
    
    
    
    
def plot_regions_deliverable(
    nodes: list[tuple[float, float, float]],
    femtocells: np.array,
    hplds: np.array,
    associations: NDArray[np.int_],
    regions,
    title: str,
    draw_node_names: bool = False,
):
    
    fig, ax = plt.subplots(figsize=(12,8))
    legend_elements = []

    region_config = {
        "alpha": 0.3, 
        "edgecolor": 'black', 
        "linewidth": 0.5, 
        "linestyle": '-',
        "color": 'orange',
    }
    def paint_regions_femto(_regions, _ax):
        for region in _regions.values():
            if isinstance(region, (Polygon, MultiPolygon, GeometryCollection)):
                if isinstance(region, Polygon):
                    polygons = [region]
                elif isinstance(region, MultiPolygon):
                    polygons = region.geoms
                else:  # GeometryCollection
                    polygons = [g for g in region.geoms if isinstance(g, Polygon)]
                
                for poly in polygons:
                    x, y = poly.exterior.coords.xy
                    _ax.fill(x, y, **region_config)
        legend_elements.append(Line2D([0], [0], color='orange', lw=6, label='FemtoCell Region'))
    
    paint_regions_femto(regions, ax)
    


    femtos_config = {
        'color': 'black',
        's': 10,
        'marker': 'o',
    }
    def paint_femtos(_nodes, _femtos, _ax):
        for i in range(len(_nodes)):
            if _femtos[i] != 1:
                continue
            
            _ax.scatter(*_nodes[i][:2], **femtos_config)
            if draw_node_names:
                _ax.annotate(str(i), (_nodes[i][0], _nodes[i][1]), textcoords='offset points', fontsize=8)
        legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=6, label='FemtoCell Position'))
    paint_femtos(nodes, femtocells, ax)
    
    
    
    pools_config = {
        'color': 'red',
        's': 10,
        'marker': '*',
    }
    def paint_pools(_nodes, _pools, _ax):
        for i in range(len(_pools)):
            if _pools[i] != 1:
                continue
            
            _ax.scatter(*_nodes[i][:2], **pools_config)
            if draw_node_names: 
                _ax.annotate(str(i), (_nodes[i][0], _nodes[i][1]), textcoords='offset points', fontsize=8)
        legend_elements.append(Line2D([0], [0], marker='*', color='w', markerfacecolor='red', markersize=10, label='PoF Pool Position'))

    paint_pools(nodes, hplds, ax)
            
    
    
    ax.axis('off')
    ax.legend(handles=legend_elements, loc='best')
    fig.tight_layout()
    
    return fig, ax
            
    
                        
                    