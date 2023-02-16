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
    # fig, ax = plt.subplots()
    fig = plt.figure()
    ax = fig.gca()
    plt.axis([-2, 2, -2, 2])

    theta = np.linspace(0,2*math.pi,100)
    x1 = np.cos(theta) - 0.5
    y1 = -np.sin(theta)
    x2 = x1*0.5
    y2 = y1*0.5

    _polygon2 = MplPolygon(np.column_stack((x1, y1)), facecolor=np.random.rand(3), alpha=0.5, edgecolor=None)
    _polygon1 = MplPolygon(np.column_stack((x2,y2)), facecolor=np.random.rand(3), alpha=0.5, edgecolor=None)
    ax.add_patch(_polygon2)
    plt.pause(2)
    ax.add_patch(_polygon1)
    plt.pause(2)

    _Reg1 = Polygon(np.column_stack((x1, y1)))
    _Reg2 = Polygon(np.column_stack((x2, y2)))

    difference = _Reg1.intersection(_Reg2)  # or difference = polygon2 - polygon1
    xx, yy = difference.exterior.coords.xy
    RegionX = xx.tolist()                    
    RegionY = yy.tolist()
    _polygon = MplPolygon(np.column_stack((RegionX, RegionY)), facecolor=np.random.rand(3), alpha=1, edgecolor=None)
    ax.add_patch(_polygon)
    print('previous pause')
    plt.pause(2)

    # polygon1 = Polygon([(0, 1), (0, 0), (1, 0), (1, 1)])
    # xx, yy = polygon1.exterior.coords.xy
    # RegionX = xx.tolist()                    
    # RegionY = yy.tolist()
    # _polygon = MplPolygon(np.column_stack((RegionX, RegionY)), facecolor=np.random.rand(3), alpha=0.5, edgecolor=None)
    # ax.add_patch(_polygon)
    # plt.pause(2)

    # polygon2 = Polygon([(0.75, 0.25), (1.25, 0.25), (1.25, 0.75), (0.75, 0.75)])
    # xx, yy = polygon2.exterior.coords.xy
    # RegionX = xx.tolist()                    
    # RegionY = yy.tolist()
    # _polygon = MplPolygon(np.column_stack((RegionX, RegionY)), facecolor=np.random.rand(3), alpha=0.5, edgecolor=None)
    # ax.add_patch(_polygon)
    # plt.pause(2)


    # difference = polygon2.difference(polygon1)  # or difference = polygon2 - polygon1
    # xx, yy = difference.exterior.coords.xy
    # RegionX = xx.tolist()                    
    # RegionY = yy.tolist()
    # _polygon = MplPolygon(np.column_stack((RegionX, RegionY)), facecolor=np.random.rand(3), alpha=0.5, edgecolor=None)
    # ax.add_patch(_polygon)
    # plt.pause(2)

    # print('done!')


    print('TBD!')

if __name__ == '__main__':
    main()