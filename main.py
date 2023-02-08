__author__ = "Gabriel Otero Perez (gaoterop@it.uc3m.es), Jose-Manuel Martinez-Caro (jmmartinez@e-lighthouse.com)"
__credits__ = ["Gabriel Otero Perez", "Jose-Manuel Martinez-Caro"]
__version__ = "1.0.1"
__maintainer__ = "Jose-Manuel Martinez-Caro"
__email__ = "jmmartinez@e-lighthouse.com"
__status__ = "Working on"

# main.py
from bcolors import bcolors
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
import map_utils
from test import get_from_geometry_collection, get_from_multi_polygon

from shapely.geometry import Polygon, GeometryCollection, MultiPolygon, Point

def main():
    # Your program goes here
    try:
        #os.system('clear')  #clear the console before start
        os.system('cls')
        os.system('pip install openpyxl matplotlib shapely')

    except Exception as e:
        print(bcolors.FAIL + 'Error installing the required dependencies' + bcolors.ENDC)
        print(e)


    try:
        trasmitting_powers = pd.read_excel('inputParameters.xlsx','TransmittingPowers',index_col=0, header=0)
        fading_rayleigh_distribution = pd.read_excel('inputParameters.xlsx','FadingRayleighDistribution',index_col=0, header=0)
        # print(trasmitting_powers)
        # print(fading_rayleigh_distribution)

    except Exception as e:
        print(bcolors.FAIL + 'Error reading from inputParameters.xlsx file' + bcolors.ENDC)
        print(e)

    # try:
    #     WeightsTier1 = np.ones((1, fading_rayleigh_distribution.loc['NMacroCells','value']))*trasmitting_powers.loc['PMacroCells','value']
    #     WeightsTier2 = np.ones((1, fading_rayleigh_distribution.loc['NFemtoCells','value']))*trasmitting_powers.loc['PFemtoCells','value']

    #     BaseStations = np.zeros((fading_rayleigh_distribution.loc['NMacroCells','value'] + fading_rayleigh_distribution.loc['NFemtoCells','value'], 3));
        
    #     # Settle Macro cells 
    #     BaseStations[0:fading_rayleigh_distribution.loc['NMacroCells','value'],0] = fading_rayleigh_distribution.loc['Maplimit','value'] * np.random.uniform(size=fading_rayleigh_distribution.loc['NMacroCells','value'], low=1, high=fading_rayleigh_distribution.loc['NMacroCells','value'])
    #     BaseStations[0:fading_rayleigh_distribution.loc['NMacroCells','value'],1] = fading_rayleigh_distribution.loc['Maplimit','value'] * np.random.uniform(size=fading_rayleigh_distribution.loc['NMacroCells','value'], low=1, high=fading_rayleigh_distribution.loc['NMacroCells','value'])
    #     BaseStations[0:fading_rayleigh_distribution.loc['NMacroCells','value'],2] = WeightsTier1

    #     BaseStations[fading_rayleigh_distribution.loc['NMacroCells','value']:,0] = fading_rayleigh_distribution.loc['Maplimit','value'] * np.random.uniform(size=fading_rayleigh_distribution.loc['NFemtoCells','value'], low=1, high=fading_rayleigh_distribution.loc['NFemtoCells','value'])
    #     BaseStations[fading_rayleigh_distribution.loc['NMacroCells','value']:,1] = fading_rayleigh_distribution.loc['Maplimit','value'] * np.random.uniform(size=fading_rayleigh_distribution.loc['NFemtoCells','value'], low=1, high=fading_rayleigh_distribution.loc['NFemtoCells','value'])

    #     # print(BaseStations)

    #     Stations = BaseStations.shape
    #     Npoints = Stations[0] #actually here
    # except Exception as e:
    #     print(bcolors.FAIL + 'Error calculating intermediate variables' + bcolors.ENDC)
    #     print(e)

    try:
        BaseStations = pd.read_excel('inputParameters.xlsx','nice_setup',index_col=None, header=None).to_numpy()
        Stations = BaseStations.shape
        Npoints = Stations[0] 
        print(Stations, Npoints)

    except Exception as e:
        print(bcolors.FAIL + 'Error importing the nice_setup sheet' + bcolors.ENDC)
        print(e)

    try:
        colorsBS = np.zeros((Npoints, 3))
        fig, ax = plt.subplots()
        plt.axis([0, fading_rayleigh_distribution.loc['Maplimit','value'], 0, fading_rayleigh_distribution.loc['Maplimit','value']])
        for a in range(0,Npoints):
            colorsBS[a] = np.random.uniform(size=3, low=0, high=1)
            ax.plot(BaseStations[a,0], BaseStations[a,1], 'o',color = colorsBS[a])
            ax.text(BaseStations[a,0], BaseStations[a,1], 'P'+str(a) , ha='center', va='bottom')

        plt.show(block=False)

    except Exception as e:
        print(bcolors.FAIL + 'Error importing the printing the BSs' + bcolors.ENDC)
        print(e)

    WholeRegionX = [0, 0, fading_rayleigh_distribution.loc['Maplimit','value'], fading_rayleigh_distribution.loc['Maplimit','value']]
    WholeRegionY = [0, fading_rayleigh_distribution.loc['Maplimit','value'], fading_rayleigh_distribution.loc['Maplimit','value'], 0]
    UnsoldRegionX = WholeRegionX;
    UnsoldRegionY = WholeRegionY;

    # print(BaseStations)
    Regions = {}
    aa = False
    # print(Regions)
    fig2, ax2 = plt.subplots()
    # fig3, ax3 = plt.subplots()
    plt.show(block=False)
    
    _polygon = MplPolygon(np.column_stack((UnsoldRegionX, UnsoldRegionY)), facecolor=np.random.rand(3), alpha=0.5, edgecolor=None)
    # ax2.add_patch(_polygon)
    for k in range(Npoints-1,-1,-1):
        print('-- k: ' + str(k))
        RegionX = UnsoldRegionX
        RegionY = UnsoldRegionY
        # if k != 22:
        #     ax2.plot(UnsoldRegionX,UnsoldRegionY)
        col = np.random.rand(3)
        for j in range(0,Npoints):
            # y, x = 1
            # print(y)
            # print(x)
            _Reg = _Reg1T= _Reg2T = _Reg1L = _Reg2L = _Reg1F = _Reg2F = _RegF = _RegT = _RegL = None
            if (j<k):

                if(BaseStations[k,2] != BaseStations[j,2]):
                    _resp = map_utils.apollonius_circle_path_loss(BaseStations[k][:2], BaseStations[j][:2], BaseStations[k][2], BaseStations[j][2], trasmitting_powers.loc['alpha_loss','value'])
                    _Circ = map_utils.get_circle(_resp[0], _resp[1], _resp[2])

                    _Reg1T = Polygon(np.column_stack((RegionX, RegionY)))
                    _Reg2T = Polygon(np.column_stack((_Circ[0], _Circ[1])))

                    _RegT = _Reg1T.intersection(_Reg2T)
                    if type(_RegT) == GeometryCollection: 
                        print('----- k: ' + str(k))
                        _RegT = _RegT.convex_hull
                        # _RegT = get_from_geometry_collection(_RegT)
                    
                    xx, yy = _RegT.exterior.coords.xy
                    RegionX = xx.tolist()                    
                    RegionY = yy.tolist()
                else:
                    _R = np.array(map_utils.get_dominance_area(BaseStations[k][:2], BaseStations[j][:2], fading_rayleigh_distribution.loc['Maplimit','value']))
                    _Reg1L = Polygon(np.column_stack((RegionX, RegionY)))
                    _Reg2L = Polygon(np.column_stack((_R[0], _R[1])))

                    _RegL = _Reg1L.intersection(_Reg2L)
                    xx, yy = _RegL.exterior.coords.xy
                    RegionX = xx.tolist()                    
                    RegionY = yy.tolist()

                # _polygon = MplPolygon(np.column_stack((RegionX, RegionY)), facecolor=col, alpha=0.5, edgecolor=None)
                # ax2.add_patch(_polygon)
                # fig2.canvas.draw()
                # fig2.canvas.flush_events()
        Regions[k] = [RegionX, RegionY]

        _polygon = MplPolygon(np.column_stack((RegionX, RegionY)), facecolor=np.random.rand(3), alpha=0.5, edgecolor=None)
        ax.add_patch(_polygon)

        _Reg1F = Polygon(np.column_stack((RegionX, RegionY)))
        _Reg2F = Polygon(np.column_stack((UnsoldRegionX, UnsoldRegionY)))

        _RegF = _Reg2F.difference(_Reg1F)
        if type(_RegF) == GeometryCollection: 
            print('----- k: ' + str(k))
            _RegF = _RegF.convex_hull
        if type(_RegF) == MultiPolygon: 
            print('----- k: ' + str(k))
            _RegF = get_from_multi_polygon(_RegF)   


        # _chP = Point(BaseStations[k,0], BaseStations[k,1])
        # print(_chP)
        # print(_RegF.contains(_chP))
        # _chP = Point(999, 999)
        # print(_RegF.contains(_chP))
        # _chP = Point(473.32, 743.16)
        # print(_RegF.contains(_chP))

        xx, yy = _RegF.exterior.coords.xy
        UnsoldRegionX = xx.tolist()                    
        UnsoldRegionY = yy.tolist()


        # ax3.plot(xx,yy)
        # print('UnsoldRegionX')
        # print(UnsoldRegionX)
        # print('UnsoldRegionY')
        # print(UnsoldRegionY)

        # # Plotting the patch
        # polygon = MplPolygon(np.column_stack((RegionX, RegionY)), facecolor=np.random.rand(3), alpha=0.35, edgecolor='none')
        # ax.add_patch(polygon)

        # # Performing the subtraction
        # UnsoldPolygon = Polygon(np.column_stack((UnsoldRegionX, UnsoldRegionY)))
        # RegionPolygon = Polygon(np.column_stack((RegionX, RegionY)))
        # UnsoldPolygon = UnsoldPolygon.difference(RegionPolygon)
        # print('------> UnsoldPolygon')
        # print(UnsoldPolygon)
        # if type(UnsoldPolygon) == MultiPolygon:
        #     print(UnsoldPolygon.convex_hull)
        #     UnsoldPolygon = UnsoldPolygon.convex_hull
        # UnsoldRegionX, UnsoldRegionY = UnsoldPolygon.exterior.xy

        # Slow down for the viewer
        plt.pause(0.25)        

    
    print ('\nTEST2')
    print (BaseStations)
    plt.show()



if __name__ == '__main__':
    main()