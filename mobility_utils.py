__author__ = "Gabriel Otero Perez (gaoterop@it.uc3m.es), Jose-Manuel Martinez-Caro (jmmartinez@e-lighthouse.com)"
__credits__ = ["Gabriel Otero Perez", "Jose-Manuel Martinez-Caro"]
__version__ = "1.0.1"
__maintainer__ = "Jose-Manuel Martinez-Caro"
__email__ = "jmmartinez@e-lighthouse.com"
__status__ = "Working on"

from random import uniform
import math
import numpy as np

s_mobility_tmp = []
nodeIndex_tmp = []
data = {
    'V_TIME': [], # (m)
    'V_POSITION_X': [], # (m)
    'V_POSITION_Y': [], # (m/s)
    'V_DIRECTION': [], # pause time (s)
    'V_SPEED_MAGNITUDE': [], # walk time(s)
    'V_IS_MOVING': [], # (degrees)
    'V_DURATION': [] # (
}

def Out_adjustDuration_random_waypoint(time,duration,dict):
    if (time+duration) >= dict['SIMULATION_TIME']:
        return dict['SIMULATION_TIME'] - time


def Out_setRestrictedWalk_random_waypoint(previousX,previousY,previousDuration,previousTime, dict):
    
    x_tmp = previousX
    y_tmp = previousY
    time_tmp = previousTime + previousDuration
    duration_tmp = Out_adjustDuration_random_waypoint(time_tmp, uniform(dict['V_WALK_INVERVAL'][0],dict['V_WALK_INVERVAL'][1]), dict)
    direction_tmp = uniform(dict['V_DIRECTION_INVERVAL'][0],dict['V_DIRECTION_INVERVAL'][1])
    speed = uniform (dict['V_SPEED_INTERVAL'][0], dict['V_SPEED_INTERVAL'][1])
    distance_tmp = speed * duration_tmp

    if distance_tmp == 0: # No movement
        s_mobility_tmp[nodeIndex_tmp]['V_TIME'] = time_tmp
        s_mobility_tmp[nodeIndex_tmp]['V_POSITION_X'] = x_tmp
        s_mobility_tmp[nodeIndex_tmp]['V_POSITION_Y'] = y_tmp
        s_mobility_tmp[nodeIndex_tmp]['V_DIRECTION'] = direction_tmp
        s_mobility_tmp[nodeIndex_tmp]['V_SPEED_MAGNITUDE'] = speed
        s_mobility_tmp[nodeIndex_tmp]['V_IS_MOVING'] = True
        s_mobility_tmp[nodeIndex_tmp]['V_DURATION'] = duration_tmp
        
    else: # The loop begins
        flag_mobility_finised = False
        while (~flag_mobility_finised):
            x_dest = x_tmp + distance_tmp * math.cos(math.radians(direction_tmp))
            y_dest = y_tmp + distance_tmp * math.sin(math.radians(direction_tmp))
            flag_mobility_was_outside = False
            if x_dest > dict['V_POSITION_X_INTERVAL'][1]:
                flag_mobility_was_outside = True
                new_direction = 180 - direction_tmp
                x_dest = dict['V_POSITION_X_INTERVAL'][1]
                y_dest = y_tmp + np.diff([x_tmp, x_dest]) * np.tan(math.radians(direction_tmp))
            if x_dest < dict['V_POSITION_X_INTERVAL'][0]:
                flag_mobility_was_outside = True
                new_direction = 180 - direction_tmp
                x_dest = dict['V_POSITION_X_INTERVAL'][0]
                y_dest = y_tmp + np.diff([x_tmp, x_dest]) * np.tan(math.radians(direction_tmp))
            if y_dest > dict['V_POSITION_Y_INTERVAL'][1]:
                flag_mobility_was_outside = True
                new_direction = -direction_tmp
                y_dest = dict['V_POSITION_Y_INTERVAL'][1]
                x_dest = x_tmp + np.diff([y_tmp, y_dest]) / np.tan(math.radians(direction_tmp))
            if y_dest < dict['V_POSITION_Y_INTERVAL'][0]:
                flag_mobility_was_outside = True
                new_direction = -direction_tmp
                y_dest = dict['V_POSITION_Y_INTERVAL'][0]
                x_dest = x_tmp + np.diff([y_tmp, y_dest]) / np.tan(math.radians(direction_tmp))
            
            current_distance = abs(np.diff([x_tmp, x_dest]) + 1j * np.diff([y_tmp, y_dest]) ) # Movement distance in a complex plane
            current_duration = Out_adjustDuration_random_waypoint(time_tmp, current_distance/speed, dict) # Duration of the movement
            
            s_mobility_tmp[nodeIndex_tmp]['V_TIME'] = time_tmp
            s_mobility_tmp[nodeIndex_tmp]['V_POSITION_X'] = x_tmp
            s_mobility_tmp[nodeIndex_tmp]['V_POSITION_Y'] = y_tmp
            s_mobility_tmp[nodeIndex_tmp]['V_DIRECTION'] = direction_tmp
            s_mobility_tmp[nodeIndex_tmp]['V_SPEED_MAGNITUDE'] = speed
            s_mobility_tmp[nodeIndex_tmp]['V_IS_MOVING'] = True
            s_mobility_tmp[nodeIndex_tmp]['V_DURATION'] = duration_tmp

            if flag_mobility_was_outside:
                time_tmp = time_tmp + current_duration
                duration_tmp = duration_tmp - current_duration
                distance_tmp = distance_tmp - current_distance
                x_tmp = x_dest; y_tmp = y_dest
                direction_tmp = new_direction
            else:
                flag_mobility_finised = True


def generate_mobility (dict):
    s_mobility_tmp = []

    for nodeIndex_tmp in range(dict['NB_NODES']):
        s_mobility_tmp[nodeIndex_tmp] = data
        previousX = uniform(dict['V_POSITION_X_INTERVAL'][0], dict['V_POSITION_X_INTERVAL'][1])
        previousY = uniform(dict['V_POSITION_Y_INTERVAL'][0], dict['V_POSITION_Y_INTERVAL'][1])
        previousDuration = 0
        previousTime = 0
        Out_setRestrictedWalk_random_waypoint(previousX=previousX, previousY=previousY, previousDuration=previousDuration, previousTime=previousTime, dict=dict)

        #########################
        # Promenade (TBD)

    print("TBD!")


