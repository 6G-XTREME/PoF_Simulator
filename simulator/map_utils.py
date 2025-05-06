__author__ = "Gabriel Otero Perez (gaoterop@it.uc3m.es), Jose-Manuel Martinez-Caro (jmmartinez@e-lighthouse.com)"
__credits__ = ["Gabriel Otero Perez", "Jose-Manuel Martinez-Caro"]
__version__ = "1.0.1"
__maintainer__ = "Jose-Manuel Martinez-Caro"
__email__ = "jmmartinez@e-lighthouse.com"
__status__ = "Validated"

import numpy as np
import sys
from math import pi, sqrt
from simulator.bcolors import bcolors
from simulator.polygon_cut import polyclip
from shapely.geometry import Point, GeometryCollection, MultiPolygon, Polygon
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

def create_regions(Npoints, BaseStations, alpha_loss, ax = None, config_parameters = {}, canvas_widget = None, polygon_bounds: list[tuple[float, float]] = None, map_size: float = 1000):
    """
    Create regions for the coverage of the base stations.
    
    Args:
        Npoints: int - Number of points to create regions for.
        BaseStations: list - List of base stations. Each base station is a tuple (x, y, p_tx).
        ax: matplotlib.axes.Axes - Axes to plot the regions on.
        alpha_loss: float - Alpha loss.
        config_parameters: dict - Configuration parameters.
        canvas_widget: matplotlib.widgets.Canvas - Canvas to plot the regions on.
        polygon_bounds: list[tuple[float, float]] - Bounds of the polygon that represents the whole region.
    """

    if polygon_bounds is None:
        polygon_bounds = [(0, 0), (0, map_size), (map_size, map_size), (map_size, 0), (0, 0)]


    _WholeRegion = Polygon(polygon_bounds)
    if not _WholeRegion.is_valid:
        _WholeRegion = _WholeRegion.buffer(0)
    _UnsoldRegion = _WholeRegion
    Regions = {}
    
    for k in range(Npoints-1,-1,-1):
        _Region = _UnsoldRegion
        for j in range(0,Npoints):
            if (j<k):
                if(BaseStations[k,2] != BaseStations[j,2]):
                    _resp = apollonius_circle_path_loss(BaseStations[k][:2], BaseStations[j][:2], BaseStations[k][2], BaseStations[j][2], alpha_loss)
                    _Circ = get_circle(_resp)

                    _Reg2 = Polygon(_Circ)
                    if not _Reg2.is_valid:
                        _Reg2 = _Reg2.buffer(0)
                    _Region = _Region.buffer(0.0001).intersection(_Reg2.buffer(0.0001))
                else:
                    _R = get_dominance_area(BaseStations[k][:2], BaseStations[j][:2])
                    if not _R.is_valid:
                        _R = _R.buffer(0)
                    _Region = _Region.buffer(0.0001).intersection(_R.buffer(0.0001))

            Regions[k] = _Region
        
        if ax is not None:
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
        if config_parameters.get('show_plots', False):
            if canvas_widget is None:
                plt.pause(config_parameters.get('speed_live_plots', 0.01))
        if canvas_widget is not None: 
            canvas_widget.draw()
    return Regions

def apollonius_circle_path_loss (P1, P2, w1, w2, alpha):
    """
    Apollonius circle path loss is a function that calculates the apollonius circle for two points and two weights.
    Args:
        P1: tuple[float, float] - First point
        P2: tuple[float, float] - Second point
        w1: float - First weight
        w2: float - Second weight
        alpha: float - Alpha
    Returns:
        tuple[float, float, float] - Center (x, y) and radius of the apollonius circle
    """
    try:
        _lambda = (w1/w2)**(1/alpha)
        _lambda = _lambda.real

        _Cx = (P1[0] - P2[0]*_lambda**2)/(1-_lambda**2)
        _Cy = (P1[1] - P2[1]*_lambda**2)/(1-_lambda**2)

        _r = _lambda * sqrt((P1[0] - P2[0])**2 + (P1[1] -P2[1])**2) /  np.linalg.norm(1 - _lambda**2)

        return _Cx, _Cy, _r

    except Exception as e:
        print(bcolors.FAIL + 'Error in function: ' + sys._getframe( ).f_code.co_name + bcolors.ENDC)
        print(bcolors.FAIL + 'Error in file: '+ sys._getframe( ).f_code.co_filename + bcolors.ENDC)
        print(e)


def get_circle(var):
    try:
        _aux = pi/50
        _th = np.arange(0,(2*pi)+_aux,_aux)

        xunit = var[2] * np.cos(_th) + var[0]
        yunit = var[2] * np.sin(_th) + var[1]

        res = [(x, y) for x, y in zip(xunit, yunit)]
        return res
    except Exception as e:
        print(bcolors.FAIL + 'Error in function: ' + sys._getframe( ).f_code.co_name + bcolors.ENDC)
        print(bcolors.FAIL + 'Error in file: '+ sys._getframe( ).f_code.co_filename + bcolors.ENDC)
        print(e)

def get_dominance_area(P1, P2):
    _medZero, _medOne = perpendicular_bisector(P1, P2)

    _WholeRegion = Polygon([(0,0), (0,1000), (1000,1000), (1000, 0)])
    
    _c =polyclip(_WholeRegion, [0, _medZero], [1, _medOne])

    _point = Point(P1[0], P1[1])    
    _polygon = Polygon(_c)
    
    
    if(_polygon.contains(_point) == False):
        _Reg1 = Polygon(_WholeRegion)
        
        _Reg = _Reg1.difference(_polygon)
        # xx, yy = _Reg.exterior.coords.xy
        # _a = xx.tolist()                    
        # _b = yy.tolist()
        return _Reg
    else:
        # _a = _c[:,0]
        # _b = _c[:,1]
        return _polygon

def get_euclidean_distance(X, Y):
    return sqrt((X[0]-Y[0])**2 + (X[1]-Y[1])**2)

def get_distance_in_kilometers(X, Y, scale):
    # Calculate Euclidean distance in map units
    d_units = get_euclidean_distance(X, Y)

    # Convert distance to kilometers, based on the provided scale
    return d_units / scale

def perpendicular_bisector(P1, P2):
    _xmed = (P1[0] + P2[0])/2
    _ymed = (P1[1] + P2[1])/2

    _a = -1/((P2[1] - P1[1])/(P2[0] - P1[0]))
    _b = _ymed - (_a*_xmed)

    return _b, (_a + _b)

def search_closest_bs(P, Regions):
    # Regions are sorted from lowest to highest preference or weight.
    closest = -1 # Return negative so it does not confuses with node 0
                 # If not found, default value should be -1

    for l in range(len(Regions)):
        if isinstance(Regions[l], Polygon):
            polygon = Regions[l]
            if polygon.contains(Point(P)):
                closest = l
        # Undetermined case that a region is a MultiPolygon... 
        elif isinstance(Regions[l], MultiPolygon):
            multipolygon = Regions[l]
            poly = multipolygon.envelope
            if poly.contains(Point(P)):
                closest = l
        # Undetermined case that a region is a GeometryCollection...
        elif isinstance(Regions[l], GeometryCollection):
            poly = Regions[l].convex_hull
            if poly.contains(Point(P)):
                closest = l

    return closest

def search_closest_bs_optimized(P, Regions, BaseStations, NMacroCells):
    # Regions are sorted from lowest to highest preference or weight.
    femtos = BaseStations[NMacroCells:] # (NMacrocells - len(BaseSations))
    macros = BaseStations[:NMacroCells] # (0 - NMacroCells)
    
    # Optimized algorithm for finding the best BS based on priority
    # 1. Find the closest Femtocell to the user, based on distance
    # 2. Check if the user is within the femtocell area
    #    2a. User within area -> return that femtocell
    #    2b. User not within area -> find, and return, closest macrocell
    
    closest_bs = -1
    lowest_dist = float('inf')
    
   
    # 1. Find the closest Femtocell to the user, based on distance
    for i, femto in enumerate(femtos):
        distance = get_euclidean_distance(P, femto)
        if distance < lowest_dist:
            lowest_dist = distance
            closest_bs = i + NMacroCells # Need to correct the index, femtos real index start at NMacroCells
   
    # 2. Check if the user is within the femtocell area
    selected_region = Regions[closest_bs]
    if isinstance(selected_region, Polygon):
        if selected_region.contains(Point(P)):
            #    2a. User within area -> return that femtocell
            return closest_bs
    # Undetermined case that a region is a MultiPolygon... 
    elif isinstance(selected_region, MultiPolygon):
        poly = selected_region.envelope
        if poly.contains(Point(P)):
            #    2a. User within area -> return that femtocell
            return closest_bs
    # Undetermined case that a region is a GeometryCollection...
    elif isinstance(selected_region, GeometryCollection):
        poly = selected_region.convex_hull
        if poly.contains(Point(P)):
            #    2a. User within area -> return that femtocell
            return closest_bs
    
    #    2b. User not within area -> find, and return, closest macrocell  
    return search_closest_macro(P, macros) # no need to correct the index, macros start at index 0


def search_closest_macro(Device, BaseStations):
    temp_dist = float('inf')
    closestMacro = None

    for station in range(len(BaseStations)):
        temp = get_euclidean_distance(Device, BaseStations[station, :])
        if temp < temp_dist:
            temp_dist = temp
            closestMacro = station
    
    return closestMacro