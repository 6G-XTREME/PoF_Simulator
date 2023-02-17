import numpy as np
import matplotlib.pyplot as plt
#from matplotlib.patches import Polygon
from shapely.geometry import Polygon, GeometryCollection, MultiPolygon, LineString
from matplotlib.patches import Polygon as MplPolygon
import math


def get_from_multi_polygon(_Reg):
    mycoordslist = [list(x.exterior.coords) for x in _Reg.geoms]
    _res = []
    for i in mycoordslist:
        for j in i:
            _res.append(j)
    _Reg = Polygon(_res)
    return _Reg

def get_from_geometry_collection(_Reg):
    c = 0
    _res = []
    # print(_Reg)
    for i in _Reg.geoms:
        print(i)
        if type(i) == Polygon: 
            for j in i.exterior.coords:
                _res.append(j)
            if hasattr(i, 'interior'):
                for j in i.interior.coords:
                    _res.append(j)
        # if type(i) == LineString:
        #     # print(i)
        #     # print(i.coords)
        #     xx, yy = i.coords.xy
        #     for j in range(len(xx)):
        #         _res.append((xx[j], yy[j]))
    _Reg = Polygon(_res)
    return _Reg

def main ():
    # Creating an empty dictionary
    myDict = {}
    
    # Adding list as value
    myDict["key1"] = [1, 2]
    
    # creating a list
    lst = ['Geeks', 'For', 'Geeks']
    
    # Adding this list as sublist in myDict
    myDict["key1"].append(lst)
    
    print(myDict)
    print(myDict['key1'][2])

    data = {
        'V_TIME':  [None] *10, # (m)
        'V_POSITION_X':  [[]] *10, # (m)
        'V_POSITION_Y':  [[]] *10, # (m/s)
        'V_DIRECTION':  [[]] *10, # pause time (s)
        'V_SPEED_MAGNITUDE':  [[]] *10, # walk time(s)
        'V_IS_MOVING':  [[]] *10, # (degrees)
        'V_DURATION':  [[]] *10 # (
    }

    # data['V_TIME'][0].append(10.225)
    print(data['V_TIME'][0])
    if data['V_TIME'][0] == None: data['V_TIME'][0] = [0]
    else: data['V_TIME'][0].append(10)
    if data['V_TIME'][0] == None: data['V_TIME'][0] = [0]
    else: data['V_TIME'][0].append(10)
    print(data)

    print('TBD!')

if __name__ == '__main__':
    main()