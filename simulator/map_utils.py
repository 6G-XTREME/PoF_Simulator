__author__ = "Gabriel Otero Perez (gaoterop@it.uc3m.es), Jose-Manuel Martinez-Caro (jmmartinez@e-lighthouse.com)"
__credits__ = ["Gabriel Otero Perez", "Jose-Manuel Martinez-Caro"]
__version__ = "1.0.1"
__maintainer__ = "Jose-Manuel Martinez-Caro"
__email__ = "jmmartinez@e-lighthouse.com"
__status__ = "Working on"

import numpy as np
import sys
from math import pi, sqrt
from simulator.bcolors import bcolors
from simulator.polygon_cut import polyclip
from shapely.geometry import Point, GeometryCollection, MultiPolygon, Polygon

def apollonius_circle_path_loss (P1, P2, w1, w2, alpha):
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
    
    return _a, _b


def get_euclidean_distance(X, Y):
    return sqrt((X[0]-Y[0])**2 + (X[1]-Y[1])**2)

def perpendicular_bisector(P1, P2):
    _xmed = (P1[0] + P2[0])/2
    _ymed = (P1[1] + P2[1])/2
    _med = (_xmed, _ymed)

    _a = -1/((P2[1] - P1[1])/(P2[0] - P1[0]))
    _b = _ymed - (_a*_xmed)

    return _b, (_a + _b)

def search_closest_bs(P, Regions):
    # Regions are sorted from lowest to highest preference or weight.
    closest = 0

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
