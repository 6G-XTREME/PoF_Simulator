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
import map_utils

from shapely.geometry import Polygon

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
        fig = plt.figure()
        plt.axis([0, fading_rayleigh_distribution.loc['Maplimit','value'], 0, fading_rayleigh_distribution.loc['Maplimit','value']])
        for a in range(0,Npoints):
            colorsBS[a] = np.random.uniform(size=3, low=0, high=1)
            plt.plot(BaseStations[a,0], BaseStations[a,1], 'o',color = colorsBS[a])
            plt.text(BaseStations[a,0], BaseStations[a,1], 'P'+str(a+1) , ha='center', va='bottom')

        plt.show(block=False)

    except Exception as e:
        print(bcolors.FAIL + 'Error importing the printing the BSs' + bcolors.ENDC)
        print(e)

    WholeRegionX = [0, 0, fading_rayleigh_distribution.loc['Maplimit','value'], fading_rayleigh_distribution.loc['Maplimit','value']]
    WholeRegionY = [0, fading_rayleigh_distribution.loc['Maplimit','value'], fading_rayleigh_distribution.loc['Maplimit','value'], 0]
    UnsoldRegionX = WholeRegionX;
    UnsoldRegionY = WholeRegionY;

    print(BaseStations)
    Regions = {}
    # print(Regions)

    for k in range(Npoints-1,0,-1):
        RegionX = UnsoldRegionX
        RegionY = UnsoldRegionY
        for j in range(0,Npoints):
            print('k: ' + str(k) + ' | j: '+ str(j))
            print('RegionX')
            print(RegionX)
            print('RegionY')
            print(RegionY)
            if (j<k):

                if(BaseStations[k,2] != BaseStations[j,2]):
                    print('DENTRO IF')
                    # print('k: ' + str(k) + ' | j: '+ str(j))
                    # print('BaseStations(22,(0,1): ')
                    # print(BaseStations[22,(0,1)])
                    # print('BaseStations(2,(0,1): ')
                    # print(BaseStations[2,(0,1)])

                    _resp = map_utils.apollonius_circle_path_loss(BaseStations[k][:2], BaseStations[j][:2], BaseStations[k][2], BaseStations[j][2], trasmitting_powers.loc['alpha_loss','value'])
                    # print('_resp')
                    # print(_resp)
                    _Circ = map_utils.get_circle(_resp[0], _resp[1], _resp[2])
                    print('_Circ')
                    print(_Circ)

                    _Reg1 = Polygon((RegionX[i], RegionY[i]) for i in range(0, len(RegionX)))
                    _Reg2 = Polygon((_Circ[0][i], _Circ[1][i]) for i in range(0, len(_Circ[0])))

                    print('_Reg1')
                    print(_Reg1)
                    print('2')
                    print(_Reg2)

                    print(_Reg1.intersects(_Reg2))
                    _Reg = _Reg1.intersection(_Reg2)
                    print('_Reg')
                    print(_Reg)
                    xx, yy = _Reg.exterior.coords.xy
                    RegionX = xx.tolist()                    
                    RegionY = yy.tolist()
                else:
                    print('DENTRO ELSE')
                    _R = np.array(map_utils.get_dominance_area(BaseStations[k][:2], BaseStations[j][:2], fading_rayleigh_distribution.loc['Maplimit','value']))
                    print('_R')
                    print(_R)
                    _Reg1 = Polygon((RegionX[i], RegionY[i]) for i in range(0, len(RegionX)))
                    _Reg2 = Polygon((_R[0][i], _R[1][i]) for i in range(_R.shape[1]))
                    print('_Reg1')
                    print(_Reg1)
                    print('_Reg2')
                    print(_Reg2)
                    print(_Reg1.intersects(_Reg2))

                    _Reg = _Reg1.intersection(_Reg2)
                    print(_Reg)
                    xx, yy = _Reg.exterior.coords.xy
                    RegionX = xx.tolist()                    
                    RegionY = yy.tolist()

            print(RegionX)
            print(RegionY)
        Regions[k] = [RegionX, RegionY]
    print(Regions)
        

    
    print ('\nTEST2')
    print (BaseStations)
    plt.show()



if __name__ == '__main__':
    main()