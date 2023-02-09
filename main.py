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
    # try:
    #     #os.system('clear')  #clear the console before start
    #     os.system('cls')
    #     os.system('pip install openpyxl matplotlib shapely')

    # except Exception as e:
    #     print(bcolors.FAIL + 'Error installing the required dependencies' + bcolors.ENDC)
    #     print(e)


    try:
        trasmitting_powers = pd.read_excel('PoF_Simulation_PYTHON/inputParameters.xlsx','TransmittingPowers',index_col=0, header=0)
        fading_rayleigh_distribution = pd.read_excel('PoF_Simulation_PYTHON/inputParameters.xlsx','FadingRayleighDistribution',index_col=0, header=0)
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
        BaseStations = pd.read_excel('PoF_Simulation_PYTHON/inputParameters.xlsx','nice_setup',index_col=None, header=None).to_numpy()
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

    try:
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
        for k in range(Npoints-1,-1,-1):
            print('-- k: ' + str(k))
            RegionX = UnsoldRegionX
            RegionY = UnsoldRegionY
            col = np.random.rand(3)
            for j in range(0,Npoints):
                _Reg = _Reg1T= _Reg2T = _Reg1L = _Reg2L = _Reg1F = _Reg2F = _RegF = _RegT = _RegL = None
                if (j<k):

                    if(BaseStations[k,2] != BaseStations[j,2]):
                        _resp = map_utils.apollonius_circle_path_loss(BaseStations[k][:2], BaseStations[j][:2], BaseStations[k][2], BaseStations[j][2], trasmitting_powers.loc['alpha_loss','value'])
                        _Circ = map_utils.get_circle(_resp[0], _resp[1], _resp[2])

                        _Reg1 = Polygon(np.column_stack((RegionX, RegionY)))
                        _Reg2 = Polygon(np.column_stack((_Circ[0], _Circ[1])))

                        _Reg = _Reg1.intersection(_Reg2)
                        if type(_Reg) == GeometryCollection: 
                            print('----- k: ' + str(k))
                            _Reg = _Reg.convex_hull
                            # _RegT = get_from_geometry_collection(_RegT)
                        
                        
                        xx, yy = _Reg.exterior.coords.xy
                        RegionX = xx.tolist()                    
                        RegionY = yy.tolist()
                    else:
                        _R = np.array(map_utils.get_dominance_area(BaseStations[k][:2], BaseStations[j][:2], fading_rayleigh_distribution.loc['Maplimit','value']))
                        _Reg1 = Polygon(np.column_stack((RegionX, RegionY)))
                        _Reg2 = Polygon(np.column_stack((_R[0], _R[1])))

                        _Reg = _Reg1.intersection(_Reg2)
                        xx, yy = _Reg.exterior.coords.xy
                        RegionX = xx.tolist()                    
                        RegionY = yy.tolist()

            Regions[k] = [RegionX, RegionY]

            _polygon = MplPolygon(np.column_stack((RegionX, RegionY)), facecolor=np.random.rand(3), alpha=0.5, edgecolor=None)
            ax.add_patch(_polygon)

            _Reg1 = Polygon(np.column_stack((RegionX, RegionY)))
            _Reg2 = Polygon(np.column_stack((UnsoldRegionX, UnsoldRegionY)))

            _Reg = _Reg2.difference(_Reg1)
            if type(_Reg) == GeometryCollection: 
                print('----- k: ' + str(k))
                _Reg = _Reg.convex_hull
            if type(_Reg) == MultiPolygon: 
                print('----- k: ' + str(k))
                _Reg = get_from_multi_polygon(_Reg)   

            xx, yy = _Reg.exterior.coords.xy
            UnsoldRegionX = xx.tolist()                    
            UnsoldRegionY = yy.tolist()

            # # Plotting the patch
            polygon = MplPolygon(np.column_stack((UnsoldRegionX, UnsoldRegionY)), facecolor=np.random.rand(3), alpha=0.35, edgecolor='none')
            ax2.add_patch(polygon)

            # Slow down for the viewer
            plt.pause(0.25)    
    except Exception as e:
        print(bcolors.FAIL + 'Error plotting the BSs coverage' + bcolors.ENDC)
        print(e)    

    sim_input = {
        'V_POSITION_X_INTERVAL': [0, fading_rayleigh_distribution.loc['Maplimit','value']], # (m)
        'V_POSITION_Y_INTERVAL': [0, fading_rayleigh_distribution.loc['Maplimit','value']], # (m)
        'V_SPEED_INTERVAL': [1, 10], # (m/s)
        'V_PAUSE_INTERVAL': [0, 3], # pause time (s)
        'V_WALK_INTERVAL': [30.00, 60.00], # walk time(s)
        'V_DIRECTION_INTERVAL': [-180, 180], # (degrees)
        'SIMULATION_TIME': fading_rayleigh_distribution.loc['Simulation_Time','value'], # (s)
        'NB_NODES': fading_rayleigh_distribution.loc['Users','value'] # (
    }
    print(sim_input['SIMULATION_TIME'])


if __name__ == '__main__':
    main()