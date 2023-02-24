import numpy as np
import matplotlib.pyplot as plt
#from matplotlib.patches import Polygon
from matplotlib.patches import Polygon as MplPolygon
import math

    # Creating an empty dictionary
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString, Polygon, GeometryCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle


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

    polygon = Polygon([(0, 0), (0, 5), (5, 5), (5, 0), (0, 0)])

    # Define a geometry collection with polygons and linestrings
    geoms = [
        Polygon([(1, 1), (1, 2), (2, 2), (2, 1), (1, 1)]),
        LineString([(2, 2), (3, 3), (4, 2)]),
        Polygon([(3, 1), (4, 1), (4, 2), (3, 2), (3, 1)])
    ]

    geom_collection = GeometryCollection(geoms)

    # Calculate the intersection between the polygon and the geometry collection
    intersection = geom_collection.intersection(polygon)

    print(intersection)
    plt.show()

    print('TBD!')

if __name__ == '__main__':
    main()