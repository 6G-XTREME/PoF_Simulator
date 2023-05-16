__author__ = "Gabriel Otero Perez (gaoterop@it.uc3m.es), Jose-Manuel Martinez-Caro (jmmartinez@e-lighthouse.com)"
__credits__ = ["Gabriel Otero Perez", "Jose-Manuel Martinez-Caro"]
__version__ = "1.0.1"
__maintainer__ = "Jose-Manuel Martinez-Caro"
__email__ = "jmmartinez@e-lighthouse.com"
__status__ = "Validated"

import simulator.map_utils

def search_closest_macro (Device, BaseStations):
    temp_dist = float('inf')
    closestMacro = None

    for station in range(len(BaseStations)):
        temp = simulator.map_utils.get_euclidean_distance(Device, BaseStations[station, :])
        if temp < temp_dist:
            temp_dist = temp
            closestMacro = station
    
    return closestMacro
