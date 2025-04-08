from simulator.map_utils import (apollonius_circle_path_loss, get_distance_in_kilometers, get_circle, get_euclidean_distance, get_dominance_area)
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon



def create_regions(Npoints, BaseStations, ax, alpha_loss, config_parameters, canvas_widget):
    _WholeRegion = Polygon([(0 ,0), (0 ,1000), (1000 ,1000) ,(1000, 0), (0 ,0)])
    _UnsoldRegion = _WholeRegion
    Regions = {}

    for k in range(Npoints -1 ,-1 ,-1):
        _Region = _UnsoldRegion
        for j in range(0 ,Npoints):
            if ( j <k):
                if(BaseStations[k ,2] != BaseStations[j ,2]):
                    _resp = apollonius_circle_path_loss(BaseStations[k][:2], BaseStations[j][:2], BaseStations[k][2], BaseStations[j][2], alpha_loss)
                    _Circ = get_circle(_resp)

                    _Reg2 = Polygon(_Circ)
                    if not _Reg2.is_valid:
                        _Reg2 = _Reg2.buffer(0)
                    _Region = _Region.intersection(_Reg2)
                else:
                    _R = get_dominance_area(BaseStations[k][:2], BaseStations[j][:2])
                    _Region = _Region.intersection(_R)

            Regions[k] = _Region

        if isinstance(_Region, GeometryCollection):
            for geom in _Region.geoms:
                if isinstance(geom, Polygon):
                    _polygon = MplPolygon(geom.exterior.coords, facecolor=np.random.rand(3), alpha=0.5, edgecolor=None)
                    ax.add_patch(_polygon)

        elif isinstance(_Region, MultiPolygon):
            col = np.random.rand(3)
            for _Reg in _Region.geoms:
                _polygon = MplPolygon(_Reg.exterior.coords, facecolor=col, alpha=0.5, edgecolor=None)
                ax.add_patch(_polygon)
        else:
            _polygon = MplPolygon(_Region.exterior.coords, facecolor=np.random.rand(3), alpha=0.5, edgecolor=None)
            ax.add_patch(_polygon)

        _UnsoldRegion = _UnsoldRegion.difference(_Region)

        # Slow down for the viewer
        if config_parameters['show_plots']:
            if canvas_widget is None:
                plt.pause(config_parameters['speed_live_plots'])
        if canvas_widget is not None:
            canvas_widget.draw()
    return Regions