import numpy as np
import matplotlib.pyplot as plt
#from matplotlib.patches import Polygon
from shapely.geometry import Polygon, GeometryCollection, MultiPolygon, LineString
from shapely.validation import make_valid
from matplotlib.patches import Polygon as MplPolygon
import math


def main ():
    fig2, ax2 = plt.subplots()
    plt.show(block=False)

    polygon1 = Polygon([(0,0), (1, 0), (1, 1), (0, 1), (0,0)])
    # polygon2 = Polygon([(0, 0), (0.5, 0), (0.5,0.5), (0, 0.5), (0, 0)])
    # polygon2 = Polygon([(0,0),(0.1, 0.1), (0.6, 0.1), (0.6,0.6), (0.1, 0.6), (0.1, 0.1), (0,0)])
    # polygon2 = Polygon([(0.75, 0.25), (1.25, 0.25), (1.25, 0.75)])
    polygon2 = Polygon([(0.9, 0.25), (0.9, 0.75), (1.1, 0.75), (1.1, 0.25), (0.9,0.25)])
    # polygon2 = Polygon([(0.5, 0.5), (0.5, 0.25), (0.25, 0.25), (0.25, 0.5)])


    ax2.set(xlim=(-2, 2), ylim=(-2, 2))

    xx, yy = polygon1.exterior.coords.xy
    RegionX = xx.tolist()                    
    RegionY = yy.tolist()
    TotalX = RegionX
    TotalY = RegionY
    _polygon = MplPolygon(np.column_stack((RegionX, RegionY)), facecolor=np.random.rand(3), alpha=0.1, edgecolor=None)
    ax2.add_patch(_polygon)
    
    xx, yy = polygon2.exterior.coords.xy
    RegionX = xx.tolist()                    
    RegionY = yy.tolist()
    _polygon = MplPolygon(np.column_stack((RegionX, RegionY)), facecolor=np.random.rand(3), alpha=0.1, edgecolor=None)
    ax2.add_patch(_polygon)

    print(TotalX[-1])
    TotalX = TotalX[:-1] + RegionX + TotalX[-2:]
    TotalY = TotalY[:-1] + RegionY + TotalY[-2:]

    _polygon = MplPolygon(np.column_stack((TotalX, TotalY)), facecolor=np.random.rand(3), alpha=0.1, edgecolor=None)
    ax2.cla()
    ax2.set(xlim=(-2, 2), ylim=(-2, 2))
    ax2.add_patch(_polygon)

    difference = polygon1.difference(polygon2)  # or difference = polygon2 - polygon1
    print(difference)
    xx, yy = difference.exterior.coords.xy
    RegionX = xx.tolist()                    
    RegionY = yy.tolist()

    _polygon = MplPolygon(np.column_stack((RegionX, RegionY)), facecolor=np.random.rand(3), alpha=0.1, edgecolor=None)
    ax2.cla()
    ax2.set(xlim=(-2, 2), ylim=(-2, 2))
    ax2.add_patch(_polygon)
    print('TBD!')
    plt.show()

if __name__ == '__main__':
    main()