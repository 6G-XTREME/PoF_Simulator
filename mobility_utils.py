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
nodeIndex_tmp = 0
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
        s_mobility_tmp[nodeIndex_tmp]['V_TIME'].append(time_tmp)
        s_mobility_tmp[nodeIndex_tmp]['V_POSITION_X'].append(x_tmp)
        s_mobility_tmp[nodeIndex_tmp]['V_POSITION_Y'].append()
        s_mobility_tmp[nodeIndex_tmp]['V_DIRECTION'].append(direction_tmp)
        s_mobility_tmp[nodeIndex_tmp]['V_SPEED_MAGNITUDE'].append(speed)
        s_mobility_tmp[nodeIndex_tmp]['V_IS_MOVING'].append(True)
        s_mobility_tmp[nodeIndex_tmp]['V_DURATION'].append(duration_tmp)
    else: # The loop begins
        flag_mobility_finised = False
        while not flag_mobility_finised:
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
            
            s_mobility_tmp[nodeIndex_tmp]['V_TIME'].append(time_tmp)
            s_mobility_tmp[nodeIndex_tmp]['V_POSITION_X'].append(x_tmp)
            s_mobility_tmp[nodeIndex_tmp]['V_POSITION_Y'].append(y_tmp)
            s_mobility_tmp[nodeIndex_tmp]['V_DIRECTION'].append(direction_tmp)
            s_mobility_tmp[nodeIndex_tmp]['V_SPEED_MAGNITUDE'].append(speed)
            s_mobility_tmp[nodeIndex_tmp]['V_IS_MOVING'].append(True)
            s_mobility_tmp[nodeIndex_tmp]['V_DURATION'].append(duration_tmp)

            if flag_mobility_was_outside:
                time_tmp = time_tmp + current_duration
                duration_tmp = duration_tmp - current_duration
                distance_tmp = distance_tmp - current_distance
                x_tmp = x_dest; y_tmp = y_dest
                direction_tmp = new_direction
            else:
                flag_mobility_finised = True


def generate_mobility (dict):
    s_mobility = []

    for nodeIndex_tmp in range(dict['NB_NODES']):
        s_mobility_tmp[nodeIndex_tmp] = data
        previousX = uniform(dict['V_POSITION_X_INTERVAL'][0], dict['V_POSITION_X_INTERVAL'][1])
        previousY = uniform(dict['V_POSITION_Y_INTERVAL'][0], dict['V_POSITION_Y_INTERVAL'][1])
        previousDuration = 0
        previousTime = 0
        Out_setRestrictedWalk_random_waypoint(previousX=previousX, previousY=previousY, previousDuration=previousDuration, previousTime=previousTime, dict=dict)

        #########################
        # Promenade
        while s_mobility_tmp[nodeIndex_tmp]['V_TIME'][-1] < dict['SIMULATION_TIME']:
            if s_mobility_tmp[nodeIndex_tmp]['V_IS_MOVING'][-1] == False:
                previousX = s_mobility_tmp[nodeIndex_tmp]['V_POSITION_X'][-1]
                previousY = s_mobility_tmp[nodeIndex_tmp]['V_POSITION_Y'][-1]
                previousDuration = s_mobility_tmp[nodeIndex_tmp]['V_DURATION'][-1]
                previousTime = s_mobility_tmp[nodeIndex_tmp]['V_TIME'][-1]
                Out_setRestrictedWalk_random_waypoint(previousX, previousY, previousDuration, previousTime, dict)
            else:
                previousDirection = s_mobility_tmp[nodeIndex_tmp]['V_DIRECTION'][-1]
                previousSpeed = s_mobility_tmp[nodeIndex_tmp]['V_SPEED_MAGNITUDE'][-1]
                previousX = s_mobility_tmp[nodeIndex_tmp]['V_POSITION_X'][-1]
                previousY = s_mobility_tmp[nodeIndex_tmp]['V_POSITION_Y'][-1]
                previousTime = s_mobility_tmp[nodeIndex_tmp]['V_TIME'][-1]
                previousDuration = s_mobility_tmp[nodeIndex_tmp]['V_DURATION'][-1]
                distance = previousDuration * previousSpeed
                s_mobility_tmp[nodeIndex_tmp]['V_TIME'].append(previousTime + previousDuration)
                s_mobility_tmp[nodeIndex_tmp]['V_POSITION_X'].append(previousX + distance * math.cos(math.radians(previousDirection)))
                s_mobility_tmp[nodeIndex_tmp]['V_POSITION_Y'].append(previousY + distance * math.sin(math.radians(previousDirection)))
                s_mobility_tmp[nodeIndex_tmp]['V_DIRECTION'].append(0)
                s_mobility_tmp[nodeIndex_tmp]['V_SPEED_MAGNITUDE'].append(0)
                s_mobility_tmp[nodeIndex_tmp]['V_IS_MOVING'].append(False)
                s_mobility_tmp[nodeIndex_tmp]['V_DURATION'].append(Out_adjustDuration_random_waypoint(s_mobility_tmp[nodeIndex_tmp]['V_TIME'][-1], uniform(dict['V_PAUSE_INTERVAL'][0], dict['V_PAUSE_INTERVAL'][1]), dict))


        nb_speed = len(s_mobility_tmp[nodeIndex_tmp]['V_SPEED_MAGNITUDE'])
        s_mobility_tmp[nodeIndex_tmp]['V_SPEED_X'] = np.zeros((nb_speed, 1))
        s_mobility_tmp[nodeIndex_tmp]['V_SPEED_Y'] = np.zeros((nb_speed, 1))
        for s in range(nb_speed):
            speed = s_mobility_tmp[nodeIndex_tmp]['V_SPEED_MAGNITUDE'][s]
            direction = s_mobility_tmp[nodeIndex_tmp]['V_DIRECTION'][s]
            s_mobility_tmp[nodeIndex_tmp]['V_SPEED_X'][s] = speed * math.cos(math.radians(direction))
            s_mobility_tmp[nodeIndex_tmp]['V_SPEED_Y'][s] = speed * math.sin(math.radians(direction))

        v_index = s_mobility_tmp[nodeIndex_tmp]['V_DURATION'][:-1] == 0
        s_mobility_tmp[nodeIndex_tmp]['V_TIME'] = np.delete(s_mobility_tmp[nodeIndex_tmp]['V_TIME'], v_index)
        s_mobility_tmp[nodeIndex_tmp]['V_POSITION_X'] = np.delete(s_mobility_tmp[nodeIndex_tmp]['V_POSITION_X'], v_index)
        s_mobility_tmp[nodeIndex_tmp]['V_POSITION_Y'] = np.delete(s_mobility_tmp[nodeIndex_tmp]['V_POSITION_Y'], v_index)
        s_mobility_tmp[nodeIndex_tmp]['V_DIRECTION'] = np.delete(s_mobility_tmp[nodeIndex_tmp]['V_DIRECTION'], v_index)
        s_mobility_tmp[nodeIndex_tmp]['V_SPEED_MAGNITUDE'] = np.delete(s_mobility_tmp[nodeIndex_tmp]['V_SPEED_MAGNITUDE'], v_index)
        s_mobility_tmp[nodeIndex_tmp]['V_IS_MOVING'] = np.delete(s_mobility_tmp[nodeIndex_tmp]['V_IS_MOVING'], v_index)
        s_mobility_tmp[nodeIndex_tmp]['V_DURATION'] = np.delete(s_mobility_tmp[nodeIndex_tmp]['V_DURATION'], v_index)
        s_mobility_tmp[nodeIndex_tmp]['V_SPEED_X'] = np.delete(s_mobility_tmp[nodeIndex_tmp]['V_SPEED_X'],v_index)
        s_mobility_tmp[nodeIndex_tmp]['V_SPEED_Y'] = np.delete(s_mobility_tmp[nodeIndex_tmp]['V_SPEED_Y'],v_index)

        if s_mobility_tmp[nodeIndex_tmp]['V_TIME'][-1] - s_mobility_tmp[nodeIndex_tmp]['V_TIME'][-2] < 1e-14:
            s_mobility_tmp[nodeIndex_tmp]['V_TIME'].pop()
            s_mobility_tmp[nodeIndex_tmp]['V_POSITION_X'].pop()
            s_mobility_tmp[nodeIndex_tmp]['V_POSITION_Y'].pop()
            s_mobility_tmp[nodeIndex_tmp]['V_DIRECTION'].pop()
            s_mobility_tmp[nodeIndex_tmp]['V_SPEED_MAGNITUDE'].pop()
            s_mobility_tmp[nodeIndex_tmp]['V_IS_MOVING'].pop()
            s_mobility_tmp[nodeIndex_tmp]['V_DURATION'].pop()
            s_mobility_tmp[nodeIndex_tmp]['V_SPEED_X'].pop()
            s_mobility_tmp[nodeIndex_tmp]['V_SPEED_Y'].pop()

        s_mobility_tmp[nodeIndex_tmp]['V_TIME'].append(dict['SIMULATION_TIME'])
        s_mobility_tmp[nodeIndex_tmp]['V_DURATION'].append(0)
        s_mobility_tmp[nodeIndex_tmp]['V_SPEED_MAGNITUDE'].append(0)
        s_mobility_tmp[nodeIndex_tmp]['V_SPEED_X'].append(0)
        s_mobility_tmp[nodeIndex_tmp]['V_SPEED_Y'].append(0)

        node_data = {   'V_TIME':          s_mobility_tmp[nodeIndex_tmp]['V_TIME'],
                        'V_POSITION_X':    s_mobility_tmp[nodeIndex_tmp]['V_POSITION_X'],
                        'V_POSITION_Y':    s_mobility_tmp[nodeIndex_tmp]['V_POSITION_Y'],
                        'V_SPEED_X':       s_mobility_tmp[nodeIndex_tmp]['V_SPEED_X'],
                        'V_SPEED_Y':       s_mobility_tmp[nodeIndex_tmp]['V_SPEED_Y']
                    }

        s_mobility[nodeIndex_tmp] = node_data
    
    del s_mobility_tmp
    del nodeIndex_tmp
    
    return s_mobility

    print("TBD!")


