__author__ = "Gabriel Otero Perez (gaoterop@it.uc3m.es), Jose-Manuel Martinez-Caro (jmmartinez@e-lighthouse.com)"
__credits__ = ["Gabriel Otero Perez", "Jose-Manuel Martinez-Caro"]
__version__ = "1.0.1"
__maintainer__ = "Jose-Manuel Martinez-Caro"
__email__ = "jmmartinez@e-lighthouse.com"
__status__ = "Working on"

from math import pi, sqrt
import numpy as np
from bcolors import bcolors
import sys
from matplotlib import path
from polygon_cut import polyclip
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon



def apollonius_circle_path_loss (P1, P2, w1, w2, alpha):  # CHECKED
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


def get_circle(x:float, y:float, r:float): #CHECKED
    try:
        _aux = pi/50
        _th = np.arange(0,(2*pi)+_aux,_aux)

        xunit = r * np.cos(_th) + x
        yunit = r * np.sin(_th) + y

        return xunit, yunit
    except Exception as e:
        print(bcolors.FAIL + 'Error in function: ' + sys._getframe( ).f_code.co_name + bcolors.ENDC)
        print(bcolors.FAIL + 'Error in file: '+ sys._getframe( ).f_code.co_filename + bcolors.ENDC)
        print(e)

def get_dominance_area(P1, P2, limit):
    _medZero, _medOne = perpendicular_bisector(P1, P2)
    print('_med: ')
    print(_medZero)
    print(_medOne)

    _WholeRegionX = (0, 0, limit, limit)
    _WholeRegionY = (0, limit, limit, 0)
    
    aux = [_WholeRegionX, _WholeRegionY]

    _c = polyclip(np.transpose([_WholeRegionX, _WholeRegionY]), [0, _medZero], [1, _medOne])

    _point = Point(P1[0], P1[1])
    _polygon =  Polygon((_c[i,0], _c[i,1]) for i in range(0, len(_WholeRegionX)))
    
    if(_polygon.contains(_point)):
        _Reg1 = Polygon((_WholeRegionX[i], _WholeRegionY[i]) for i in range(0, len(_WholeRegionX)))
        
        _Reg = _Reg1.intersection(_polygon)
        xx, yy = _Reg.exterior.coords.xy
        _a = xx.tolist()                    
        _b = yy.tolist()
    else:
        _a = _c[:,0]
        _b = _c[:,1]
    
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
    print("TBD!")

def search_closest_bs(P, Regions):
    # Regions are sorted from lowest to highest preference or weight.
    _closest = 0
    for i in range(0, len(Regions)):
        _p = path.Path([(0,0), (0, 1), (1, 1), (1, 0)])  # square with legs length 1 and bottom left corner at the origin
        _in =_p.contains_points([(P[0], P[1])])
        _closet = _closest + _in*i
    print("Complete and review it!")
    return _closet

