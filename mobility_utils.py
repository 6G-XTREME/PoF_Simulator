__author__ = "Gabriel Otero Perez (gaoterop@it.uc3m.es), Jose-Manuel Martinez-Caro (jmmartinez@e-lighthouse.com)"
__credits__ = ["Gabriel Otero Perez", "Jose-Manuel Martinez-Caro"]
__version__ = "1.0.1"
__maintainer__ = "Jose-Manuel Martinez-Caro"
__email__ = "jmmartinez@e-lighthouse.com"
__status__ = "Working on"

from random import uniform
import math
import numpy as np

nodeIndex_tmp = 0
def Out_adjustDuration_random_waypoint(time,duration,dict):
    if (time+duration) >= dict['SIMULATION_TIME']: return dict['SIMULATION_TIME'] - time
    return duration

def add_element_to_s_mobility(variable, value):
    if s_mobility[variable][nodeIndex_tmp] == None: s_mobility[variable][nodeIndex_tmp] = [value]
    else: s_mobility[variable][nodeIndex_tmp].append(value)

def Out_setRestrictedWalk_random_waypoint(previousX, previousY, previousDuration, previousTime, dict):
    
    x_tmp = previousX
    y_tmp = previousY
    time_tmp = previousTime + previousDuration
    duration_tmp = Out_adjustDuration_random_waypoint(time_tmp, uniform(dict['V_WALK_INTERVAL'][0],dict['V_WALK_INTERVAL'][1]), dict)
    direction_tmp = uniform(dict['V_DIRECTION_INTERVAL'][0],dict['V_DIRECTION_INTERVAL'][1])
    speed = uniform (dict['V_SPEED_INTERVAL'][0], dict['V_SPEED_INTERVAL'][1])
    distance_tmp = speed * duration_tmp

    if distance_tmp == 0: # No movement
        add_element_to_s_mobility('V_TIME', time_tmp)
        add_element_to_s_mobility('V_POSITION_X', x_tmp)
        add_element_to_s_mobility('V_POSITION_Y', y_tmp)
        add_element_to_s_mobility('V_DIRECTION', direction_tmp)
        add_element_to_s_mobility('V_SPEED_MAGNITUDE', speed)
        add_element_to_s_mobility('V_IS_MOVING', True)
        add_element_to_s_mobility('V_DURATION', duration_tmp)

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
                y_dest = y_tmp + np.diff([x_tmp, x_dest])[0] * np.tan(math.radians(direction_tmp))
            if x_dest < dict['V_POSITION_X_INTERVAL'][0]:
                flag_mobility_was_outside = True
                new_direction = 180 - direction_tmp
                x_dest = dict['V_POSITION_X_INTERVAL'][0]
                y_dest = y_tmp + np.diff([x_tmp, x_dest])[0] * np.tan(math.radians(direction_tmp))
            if y_dest > dict['V_POSITION_Y_INTERVAL'][1]:
                flag_mobility_was_outside = True
                new_direction = -direction_tmp
                y_dest = dict['V_POSITION_Y_INTERVAL'][1]
                x_dest = x_tmp + np.diff([y_tmp, y_dest])[0] / np.tan(math.radians(direction_tmp))
            if y_dest < dict['V_POSITION_Y_INTERVAL'][0]:
                flag_mobility_was_outside = True
                new_direction = -direction_tmp
                y_dest = dict['V_POSITION_Y_INTERVAL'][0]
                x_dest = x_tmp + np.diff([y_tmp, y_dest])[0] / np.tan(math.radians(direction_tmp))
            
            current_distance = abs(np.diff([x_tmp, x_dest])[0] + 1j * np.diff([y_tmp, y_dest])[0] ) # Movement distance in a complex plane
            current_duration = Out_adjustDuration_random_waypoint(time_tmp, current_distance/speed, dict) # Duration of the movement
            
            add_element_to_s_mobility('V_TIME', time_tmp)
            add_element_to_s_mobility('V_POSITION_X', x_tmp)
            add_element_to_s_mobility('V_POSITION_Y', y_tmp)
            add_element_to_s_mobility('V_DIRECTION', direction_tmp)
            add_element_to_s_mobility('V_SPEED_MAGNITUDE', speed)
            add_element_to_s_mobility('V_IS_MOVING', True)
            add_element_to_s_mobility('V_DURATION', duration_tmp)

            if flag_mobility_was_outside:
                time_tmp = time_tmp + current_duration
                duration_tmp = duration_tmp - current_duration
                distance_tmp = distance_tmp - current_distance
                x_tmp = x_dest; y_tmp = y_dest
                direction_tmp = new_direction
            else:
                flag_mobility_finised = True


def generate_mobility (dict):    
    data = {
        'V_TIME':           [None] * dict['NB_NODES'], # (m)
        'V_POSITION_X':     [None] * dict['NB_NODES'], # (m)
        'V_POSITION_Y':     [None] * dict['NB_NODES'], # (m/s)
        'V_DIRECTION':      [None] * dict['NB_NODES'], # pause time (s)
        'V_SPEED_MAGNITUDE':[None] * dict['NB_NODES'], # walk time(s)
        'V_IS_MOVING':      [None] * dict['NB_NODES'], # (degrees)
        'V_DURATION':       [None] * dict['NB_NODES'],  # (
        'V_SPEED_X':        [None] * dict['NB_NODES'],  # (
        'V_SPEED_Y':        [None] * dict['NB_NODES']  # (
    }
    global s_mobility
    global nodeIndex_tmp
    s_mobility = data

    for nodeIndex_tmp in range(dict['NB_NODES']):
        # s_mobility = data
        previousX = uniform(dict['V_POSITION_X_INTERVAL'][0], dict['V_POSITION_X_INTERVAL'][1])
        previousY = uniform(dict['V_POSITION_Y_INTERVAL'][0], dict['V_POSITION_Y_INTERVAL'][1])
        previousDuration = 0
        previousTime = 0
        Out_setRestrictedWalk_random_waypoint(  previousX=previousX, 
                                                previousY=previousY, 
                                                previousDuration=previousDuration, 
                                                previousTime=previousTime, 
                                                dict=dict)

        #########################
        # Promenade
        while s_mobility['V_TIME'][nodeIndex_tmp][-1] < dict['SIMULATION_TIME']:
            if s_mobility['V_IS_MOVING'][nodeIndex_tmp][-1] == False:
                previousX = s_mobility['V_POSITION_X'][nodeIndex_tmp][-1]
                previousY = s_mobility['V_POSITION_Y'][nodeIndex_tmp][-1]
                previousDuration = s_mobility['V_DURATION'][nodeIndex_tmp][-1]
                previousTime = s_mobility['V_TIME'][nodeIndex_tmp][-1]
                Out_setRestrictedWalk_random_waypoint(previousX, previousY, previousDuration, previousTime, dict)
            else:
                previousDirection = s_mobility['V_DIRECTION'][nodeIndex_tmp][-1]
                previousSpeed = s_mobility['V_SPEED_MAGNITUDE'][nodeIndex_tmp][-1]
                previousX = s_mobility['V_POSITION_X'][nodeIndex_tmp][-1]
                previousY = s_mobility['V_POSITION_Y'][nodeIndex_tmp][-1]
                previousTime = s_mobility['V_TIME'][nodeIndex_tmp][-1]
                previousDuration = s_mobility['V_DURATION'][nodeIndex_tmp][-1]
                distance = previousDuration * previousSpeed

                add_element_to_s_mobility('V_TIME', previousTime + previousDuration)
                add_element_to_s_mobility('V_POSITION_X', previousX + distance * math.cos(math.radians(previousDirection)))
                add_element_to_s_mobility('V_POSITION_Y', previousY + distance * math.sin(math.radians(previousDirection)))
                add_element_to_s_mobility('V_DIRECTION', 0)
                add_element_to_s_mobility('V_SPEED_MAGNITUDE', 0)
                add_element_to_s_mobility('V_IS_MOVING', False)
                add_element_to_s_mobility('V_DURATION', Out_adjustDuration_random_waypoint(s_mobility['V_TIME'][nodeIndex_tmp][-1], uniform(dict['V_PAUSE_INTERVAL'][0], dict['V_PAUSE_INTERVAL'][1]), dict))

        nb_speed = len(s_mobility['V_SPEED_MAGNITUDE'][nodeIndex_tmp])
        for s in range(nb_speed):
            speed = s_mobility['V_SPEED_MAGNITUDE'][nodeIndex_tmp][s]
            direction = s_mobility['V_DIRECTION'][nodeIndex_tmp][s]
            add_element_to_s_mobility('V_SPEED_X', speed * math.cos(math.radians(direction)))
            add_element_to_s_mobility('V_SPEED_Y', speed * math.sin(math.radians(direction)))

        # To remove null pauses
        v_index = 0 in s_mobility['V_DURATION'][nodeIndex_tmp][:-1]
        if v_index:
            v_index = s_mobility['V_DURATION'][nodeIndex_tmp][:-1].index(0)
            s_mobility['V_TIME']            = np.delete(s_mobility['V_TIME'], v_index)
            s_mobility['V_POSITION_X']      = np.delete(s_mobility['V_POSITION_X'], v_index)
            s_mobility['V_POSITION_Y']      = np.delete(s_mobility['V_POSITION_Y'], v_index)
            s_mobility['V_DIRECTION']       = np.delete(s_mobility['V_DIRECTION'], v_index)
            s_mobility['V_SPEED_MAGNITUDE'] = np.delete(s_mobility['V_SPEED_MAGNITUDE'], v_index)
            s_mobility['V_IS_MOVING']       = np.delete(s_mobility['V_IS_MOVINT'], v_index)
            s_mobility['V_DURATION']        = np.delete(s_mobility['V_DURATION'], v_index)
            s_mobility['V_SPEED_X']         = np.delete(s_mobility['V_SPEED_X'], v_index)
            s_mobility['V_SPEED_Y']         = np.delete(s_mobility['V_SPEED_Y'], v_index)

        # To remove the too small difference at the end, if there is one:
        if s_mobility['V_TIME'][nodeIndex_tmp][-1] - s_mobility['V_TIME'][nodeIndex_tmp][-2] < 1e-14:
            s_mobility['V_TIME'][nodeIndex_tmp].pop()
            s_mobility['V_POSITION_X'][nodeIndex_tmp].pop()
            s_mobility['V_POSITION_Y'][nodeIndex_tmp].pop()
            s_mobility['V_DIRECTION'][nodeIndex_tmp].pop()
            s_mobility['V_SPEED_MAGNITUDE'][nodeIndex_tmp].pop()
            s_mobility['V_IS_MOVING'][nodeIndex_tmp].pop()
            s_mobility['V_DURATION'][nodeIndex_tmp].pop()
            s_mobility['V_SPEED_X'][nodeIndex_tmp].pop()
            s_mobility['V_SPEED_Y'][nodeIndex_tmp].pop()

        s_mobility['V_TIME'][nodeIndex_tmp][-1] = dict['SIMULATION_TIME']
        s_mobility['V_DURATION'][nodeIndex_tmp][-1] = 0
        s_mobility['V_SPEED_MAGNITUDE'][nodeIndex_tmp][-1] = 0
        s_mobility['V_SPEED_X'][nodeIndex_tmp][-1] = 0
        s_mobility['V_SPEED_Y'][nodeIndex_tmp][-1] = 0

        node_data = {   'V_TIME':          s_mobility['V_TIME'][nodeIndex_tmp],
                        'V_POSITION_X':    s_mobility['V_POSITION_X'][nodeIndex_tmp],
                        'V_POSITION_Y':    s_mobility['V_POSITION_Y'][nodeIndex_tmp],
                        'V_SPEED_X':       s_mobility['V_SPEED_X'][nodeIndex_tmp],
                        'V_SPEED_Y':       s_mobility['V_SPEED_Y'][nodeIndex_tmp]
                    }

        s_mobility[nodeIndex_tmp] = node_data
    
    del nodeIndex_tmp
    
    return s_mobility

    print("TBD!")


