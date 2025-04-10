from shapely.geometry import MultiPolygon, Polygon, GeometryCollection
import matplotlib.pyplot as plt
from model.NodeClass import Node



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
    