__author__ = "Gabriel Otero Perez (gaoterop@it.uc3m.es), Jose-Manuel Martinez-Caro (jmmartinez@e-lighthouse.com), Enrique Fernandez Sanchez (efernandez@e-lighthouse.com)"
__credits__ = ["Gabriel Otero Perez", "Jose-Manuel Martinez-Caro", "Enrique Fernandez Sanchez"]
__version__ = "1.0.1"
__maintainer__ = "Jose-Manuel Martinez-Caro"
__email__ = "jmmartinez@e-lighthouse.com"
__status__ = "Working on"

# main.py
from bcolors import bcolors
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from tqdm import tqdm
import map_utils, mobility_utils, user_association_utils, radio_utils
import time

from shapely.geometry import Polygon, GeometryCollection, MultiPolygon, Point, LineString

def main(live_plots = False):

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
    #     BaseStations[0:fading_rayleigh_distribution.loc['NMacroCells','value'],0] = Maplimit * np.random.uniform(size=fading_rayleigh_distribution.loc['NMacroCells','value'], low=1, high=fading_rayleigh_distribution.loc['NMacroCells','value'])
    #     BaseStations[0:fading_rayleigh_distribution.loc['NMacroCells','value'],1] = Maplimit * np.random.uniform(size=fading_rayleigh_distribution.loc['NMacroCells','value'], low=1, high=fading_rayleigh_distribution.loc['NMacroCells','value'])
    #     BaseStations[0:fading_rayleigh_distribution.loc['NMacroCells','value'],2] = WeightsTier1

    #     BaseStations[fading_rayleigh_distribution.loc['NMacroCells','value']:,0] = Maplimit * np.random.uniform(size=fading_rayleigh_distribution.loc['NFemtoCells','value'], low=1, high=fading_rayleigh_distribution.loc['NFemtoCells','value'])
    #     BaseStations[fading_rayleigh_distribution.loc['NMacroCells','value']:,1] = Maplimit * np.random.uniform(size=fading_rayleigh_distribution.loc['NFemtoCells','value'], low=1, high=fading_rayleigh_distribution.loc['NFemtoCells','value'])

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
        battery_capacity = fading_rayleigh_distribution.loc['battery_capacity','value']
        NMacroCells = fading_rayleigh_distribution.loc['NMacroCells','value']
        NFemtoCells = fading_rayleigh_distribution.loc['NFemtoCells','value']
        small_cell_consumption_ON = fading_rayleigh_distribution.loc['small_cell_consumption_ON','value']
        small_cell_consumption_SLEEP = fading_rayleigh_distribution.loc['small_cell_consumption_SLEEP','value']
        Maplimit = fading_rayleigh_distribution.loc['Maplimit','value']
        Simulation_Time = fading_rayleigh_distribution.loc['Simulation_Time','value']
        Users = fading_rayleigh_distribution.loc['Users','value']
        timeStep = fading_rayleigh_distribution.loc['timeStep','value']
        max_energy_consumption = fading_rayleigh_distribution.loc['max_energy_consumption','value']
        small_cell_current_draw = fading_rayleigh_distribution.loc['small_cell_current_draw','value']
        noise = fading_rayleigh_distribution.loc['noise','value']
        SMA_WINDOW = fading_rayleigh_distribution.loc['SMA_WINDOW','value']
        small_cell_voltage_range = 0.01 * np.array([fading_rayleigh_distribution.loc['small_cell_voltage_min','value'], fading_rayleigh_distribution.loc['small_cell_voltage_max','value']])

        PMacroCells = trasmitting_powers.loc['PMacroCells', 'value']
        PFemtoCells = trasmitting_powers.loc['PFemtoCells', 'value']
        alpha_loss = trasmitting_powers.loc['alpha_loss', 'value']
        MacroCellDownlinkBW = trasmitting_powers.loc['MacroCellDownlinkBW', 'value']
        FemtoCellDownlinkBW = trasmitting_powers.loc['FemtoCellDownlinkBW', 'value']

    except Exception as e:
        print(bcolors.FAIL + 'Error importing parameters into local variables' + bcolors.ENDC)
        print(e)

    try:
        colorsBS = np.zeros((Npoints, 3))
        fig1, ax = plt.subplots()
        plt.axis([0, Maplimit, 0, Maplimit])
        for a in range(0,Npoints):
            colorsBS[a] = np.random.uniform(size=3, low=0, high=1)
            ax.plot(BaseStations[a,0], BaseStations[a,1], 'o',color = colorsBS[a])
            ax.text(BaseStations[a,0], BaseStations[a,1], 'P'+str(a) , ha='center', va='bottom')

        plt.show(block=False)

    except Exception as e:
        print(bcolors.FAIL + 'Error importing the printing the BSs' + bcolors.ENDC)
        print(e)

    try:
        _WholeRegion = Polygon([(0,0), (0,1000), (1000,1000),(1000, 0), (0,0)])
        _UnsoldRegion = _WholeRegion

        # print(BaseStations)
        Regions = {}
        aa = False
        # print(Regions)
        #fig2, ax2 = plt.subplots()
        #plt.show(block=False)
        
        for k in range(Npoints-1,-1,-1):
            print('-- k: ' + str(k))
            _Region = _UnsoldRegion
            for j in range(0,Npoints):
                if (j<k):

                    if(BaseStations[k,2] != BaseStations[j,2]):
                        _resp = map_utils.apollonius_circle_path_loss(BaseStations[k][:2], BaseStations[j][:2], BaseStations[k][2], BaseStations[j][2], alpha_loss)
                        _Circ = map_utils.get_circle(_resp)

                        _Reg2 = Polygon(_Circ)
                        if not _Reg2.is_valid:
                            _Reg2 = _Reg2.buffer(0)
                        # if isinstance(_Region, GeometryCollection):
                        #     print('Soy GeometryCollection')
                        _Region = _Region.intersection(_Reg2)
                    else:
                        _R = map_utils.get_dominance_area(BaseStations[k][:2], BaseStations[j][:2])
                        _Region = _Region.intersection(_R)

            Regions[k] = _Region
            
            if isinstance(_Region, GeometryCollection):
                for geom in _Region.geoms:
                    if isinstance(geom, Polygon):
                        _polygon = MplPolygon(geom.exterior.coords, facecolor=np.random.rand(3), alpha=0.5, edgecolor=None)
                        ax.add_patch(_polygon)
            elif isinstance(_Region, MultiPolygon):
                col = np.random.rand(3)
                print('MultiPolygon here!')
                for _Reg in _Region.geoms:
                    _polygon = MplPolygon(_Reg.exterior.coords, facecolor=col, alpha=0.5, edgecolor=None)
                    ax.add_patch(_polygon)

            else:
                _polygon = MplPolygon(_Region.exterior.coords, facecolor=np.random.rand(3), alpha=0.5, edgecolor=None)
                ax.add_patch(_polygon)

            _UnsoldRegion = _UnsoldRegion.difference(_Region)
            

            # Slow down for the viewer
            plt.pause(0.1)    
    except Exception as e:
        print(bcolors.FAIL + 'Error plotting the BSs coverage' + bcolors.ENDC)
        print(e)    

    sim_input = {
        'V_POSITION_X_INTERVAL': [0, Maplimit], # (m)
        'V_POSITION_Y_INTERVAL': [0, Maplimit], # (m)
        'V_SPEED_INTERVAL': [1, 10], # (m/s)
        'V_PAUSE_INTERVAL': [0, 3], # pause time (s)
        'V_WALK_INTERVAL': [30.00, 60.00], # walk time(s)
        'V_DIRECTION_INTERVAL': [-180, 180], # (degrees)
        'SIMULATION_TIME': Simulation_Time, # (s)
        'NB_NODES': Users # (
    }
    print(sim_input['V_WALK_INTERVAL'])

    # Before here, validated!!! 
    
    # Generate the mobility path of users
    s_mobility = mobility_utils.generate_mobility(sim_input)
    s_mobility["NB_NODES"] = []
    for node in range(0, sim_input['NB_NODES']):
        s_mobility['NB_NODES'].append(s_mobility[node])

    sim_times = np.arange(0, sim_input['SIMULATION_TIME'] + timeStep, timeStep)

    node_list = []

    #  Create visualization plots

    for nodeIndex in range(sim_input['NB_NODES']):
        node_y = np.interp(sim_times, s_mobility['V_TIME'][nodeIndex], s_mobility['V_POSITION_Y'][nodeIndex])
        node_x = np.interp(sim_times, s_mobility['V_TIME'][nodeIndex], s_mobility['V_POSITION_X'][nodeIndex])
        node_list.append({'v_x': node_x, 'v_y': node_y})

    active_Cells = np.zeros(NMacroCells + NFemtoCells)
    node_pos_plot = []
    node_association_line = []

    for nodeIndex in range(sim_input['NB_NODES']):
        node_pos = ax.plot(node_list[nodeIndex]['v_x'][0], node_list[nodeIndex]['v_y'][0], '+', markersize=10, linewidth=2, color=[0.3, 0.3, 1])
        node_pos_plot.append(node_pos)

        closestBSDownlink = map_utils.search_closest_bs([node_list[nodeIndex]['v_x'][0], node_list[nodeIndex]['v_y'][0]], Regions)
        x = [node_list[nodeIndex]['v_x'][0], BaseStations[closestBSDownlink][0]]
        y = [node_list[nodeIndex]['v_y'][0], BaseStations[closestBSDownlink][1]]
        node_assoc, = ax.plot(x, y, color=colorsBS[closestBSDownlink])
        node_association_line.append(node_assoc)

        active_Cells[closestBSDownlink] = 1

    ax.set_title('Downlink association. Distance & Power criterion')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    text = ax.text(0, Maplimit, 'Time (sec) = 0')

    plt.show(block=False)
    
    live_smallcell_occupancy = np.zeros(len(sim_times))
    live_smallcell_occupancy[0] = sum(active_Cells[NMacroCells-1:-1])

    if live_plots:
        fig, ax = plt.subplots()
        ax.plot([0, sim_times[0]], [NFemtoCells, NFemtoCells], 'r', label='Total Small cells')
        ax.plot(sim_times[0], live_smallcell_occupancy[0], 'g', label='Small cells being used')
        ax.text(0, NFemtoCells - 1, f"Phantom Cells ON: 0")
        ax.legend()
        ax.set_title('Number of small cells under use')

    # Plot the first time slot for consumption
    live_smallcell_consumption = np.zeros(len(sim_times))
    live_smallcell_consumption[0] = live_smallcell_occupancy[0] * small_cell_consumption_ON + (NFemtoCells - live_smallcell_occupancy[0]) * small_cell_consumption_SLEEP

    if live_plots:
        fig, ax = plt.subplots()
        ax.plot([0, sim_times[0]], [small_cell_consumption_ON * NFemtoCells, small_cell_consumption_ON * NFemtoCells], 'r', label='Total always ON consumption [W]')
        ax.plot(sim_times[0], live_smallcell_consumption[0], 'g', label='Live energy consumption [W]')
        ax.text(1, small_cell_consumption_ON * NFemtoCells - 1, f"Energy consumption (Active Femtocells): 0 W")
        ax.text(1, small_cell_consumption_ON * NFemtoCells - 3, f"Energy consumption (Idle Femtocells): 0 W")
        ax.text(1, small_cell_consumption_ON * NFemtoCells - 5, f"Energy consumption (Total Femtocells): 0 W")
        ax.legend()
        ax.set_title('Live energy consumption')

    # Plot the first time slot for throughput
    live_throughput = np.zeros(len(sim_times))
    live_throughput_NO_BATTERY = np.zeros(len(sim_times))
    live_throughput_only_Macros = np.zeros(len(sim_times))

    if live_plots:
        fig, ax = plt.subplots()
        ax.plot(sim_times[0], live_throughput[0], label='With battery system')
        ax.plot(sim_times[0], live_throughput_NO_BATTERY[0], 'r--', label='Without battery system')
        ax.plot(sim_times[0], live_throughput_only_Macros[0], 'g:.', label='Only Macrocells')
        ax.legend()
        ax.set_title('Live system throughput')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Throughput [Mb/s]')

    battery_vector = battery_capacity * np.ones((1, NMacroCells + NFemtoCells))
    association_vector = np.zeros((1, len(s_mobility['NB_NODES'])))
    association_vector_overflow_alternative = np.zeros((1, len(s_mobility['NB_NODES'])))
    battery_mean_values = np.zeros(len(sim_times)) + battery_capacity

    if live_plots:
        fig, ax = plt.subplots()
        for b in range(NFemtoCells):
            ax.bar(NMacroCells + b + 1, battery_vector[0, NMacroCells + b], color='b')
        ax.set_title('Live battery state')
        
    if live_plots:
        plt.show(block=False)

    # Start the simulation!
    print("Starting simulation...")

    f = tqdm(total=100, desc='Simulating...')
    for timeIndex in range(len(sim_times)):
        # Check for clicked Cancel button
        if f.n > f.total:
            break

        # Update progress bar and message
        f.update(100 / len(sim_times))
        f.set_description("%.2f %% completed..." % (timeIndex * 100 / len(sim_times)+1))

        t = sim_times[timeIndex]
        text.set_text('Time (sec) = {:.2f}'.format(t))

        active_Cells = np.zeros(NMacroCells+NFemtoCells)
        battery_state = np.zeros(NMacroCells+NFemtoCells) # 0 = nothing; 1 = charging; 2 = discharging; 3 = discharging & charging.
        baseStation_users = np.zeros(NMacroCells+NFemtoCells) # Number of users in each base station.
        overflown_from = np.zeros(NMacroCells+NFemtoCells) # Number of users that could not be served in each BS if we had no batteries.

        for nodeIndex in range(0, len(s_mobility['NB_NODES'])):
    
            node_pos_plot[nodeIndex][0].set_data([node_list[nodeIndex]["v_x"][timeIndex], node_list[nodeIndex]["v_y"][timeIndex]])
    
            #Search serving base station
            closestBSDownlink = map_utils.search_closest_bs([node_list[nodeIndex]["v_x"][timeIndex], node_list[nodeIndex]["v_y"][timeIndex]], Regions)
            
            #If closest is a Femtocell and it is sleeping (it has no users), then, check total energy consumption
            if closestBSDownlink > NMacroCells:

                if baseStation_users[closestBSDownlink] == 0: #If inactive
                    
                    #Can I turn it on with PoF?
                    current_watts = sum(active_Cells[NMacroCells:]) * small_cell_consumption_ON + (NFemtoCells - sum(active_Cells[NMacroCells:])) * small_cell_consumption_SLEEP
                    if current_watts >= max_energy_consumption - small_cell_consumption_ON + small_cell_consumption_SLEEP: # No, I cannot. Check battery.
                        
                        #Check if we can use Femtocell's battery
                        if battery_vector[0, closestBSDownlink] > (timeStep/3600) * small_cell_current_draw:
                            X = [node_list[nodeIndex]["v_x"][timeIndex], BaseStations[closestBSDownlink, 0]]
                            Y = [node_list[nodeIndex]["v_y"][timeIndex], BaseStations[closestBSDownlink, 1]]
                            node_association_line[nodeIndex].set_data(X, Y)
                            node_association_line[nodeIndex].set_color('green')
                            node_association_line[nodeIndex].set_linestyle('--')
                            node_association_line[nodeIndex].set_linewidth(3)
                            
                            association_vector[0, nodeIndex] = closestBSDownlink # Associate.
                            
                            # Alternative if we had no batteries would be...
                            association_vector_overflow_alternative[0, nodeIndex] = user_association_utils.search_closest_macro([node_list[nodeIndex]["v_x"][timeIndex], node_list[nodeIndex]["v_y"][timeIndex]], BaseStations[0:NMacroCells, 0:2])
                            overflown_from[closestBSDownlink] += 1
                            
                            # Comment on MATLAB:
                            #active_Cells[0, closestBSDownlink] = 1 # This cell does not count for the overall PoF power budget.
                            
                            battery_state[closestBSDownlink] = 2 # Discharge battery.
                            battery_vector[0, closestBSDownlink] = max(0, battery_vector[0, closestBSDownlink] - (timeStep/3600) * small_cell_current_draw) # However, draw from Femtocell's battery.
                            baseStation_users[closestBSDownlink] += 1 # Add user.
                        else:
                            #Associate to closest Macrocell
                            closest_Macro = user_association_utils.search_closest_macro([node_list[nodeIndex]["v_x"][timeIndex], node_list[nodeIndex]["v_y"][timeIndex]], BaseStations[0:NMacroCells, 0:2])
                            X = [node_list[nodeIndex]["v_x"][timeIndex], BaseStations[closest_Macro, 0]]
                            Y = [node_list[nodeIndex]["v_y"][timeIndex], BaseStations[closest_Macro, 1]]
                            node_association_line[nodeIndex].set_data(X, Y)
                            node_association_line[nodeIndex].set_color('red')
                            node_association_line[nodeIndex].set_linestyle('--')
                            node_association_line[nodeIndex].set_linewidth(2)

                            association_vector[0, nodeIndex] = closest_Macro # Associate.
                            active_Cells[closest_Macro] = 1 
                            baseStation_users[closest_Macro] += 1 
                    else:
                        #Yes, turn on with PoF and associate
                        X = [node_list[nodeIndex]["v_x"][timeIndex], BaseStations[closestBSDownlink, 0]]
                        Y = [node_list[nodeIndex]["v_y"][timeIndex], BaseStations[closestBSDownlink, 1]]
                        node_association_line[nodeIndex].set_data(X, Y)
                        node_association_line[nodeIndex].set_color(colorsBS[closestBSDownlink])
                        node_association_line[nodeIndex].set_linestyle('-')
                        node_association_line[nodeIndex].set_linewidth(0.5)

                        association_vector[0, nodeIndex] = closestBSDownlink # Associate.
                        association_vector_overflow_alternative[0, nodeIndex] = 0 # I can use PoF. Having batteries makes no difference in this case. Alternative is not needed.
                        active_Cells[closestBSDownlink] = 1 # This cell counts for the PoF budget.
                        battery_state[closestBSDownlink] = 0 # No battery usage.
                        baseStation_users[closestBSDownlink] += 1 # Add user.
                else: # Already ON, associate to the femtocell, just add one user.
                    association_vector[0, nodeIndex] = closestBSDownlink # Associate.
                    if battery_state[closestBSDownlink] == 2.0:
                        # If we had no batteries, this user would have been gone to the closest macrocell. Search "overflow" alternative and add 1 to the "kicked" users of this femtocell in the hypothetical case we had no batteries installed. 
                        association_vector_overflow_alternative[0, nodeIndex] = user_association_utils.search_closest_macro([node_list[nodeIndex]["v_x"][timeIndex], node_list[nodeIndex]["v_y"][timeIndex]], BaseStations[0:NMacroCells, 0:2])
                        overflown_from[closestBSDownlink] += 1
                    else:
                        association_vector_overflow_alternative[0, nodeIndex] = 0
                    baseStation_users[closestBSDownlink] += 1 # Add user.

                    X = [node_list[nodeIndex]["v_x"][timeIndex], BaseStations[closestBSDownlink, 0]]
                    Y = [node_list[nodeIndex]["v_y"][timeIndex], BaseStations[closestBSDownlink, 1]]

                    if battery_state[closestBSDownlink] == 2.0:
                        # %If using battery (only check == 2 because 3 only happens later at chaging decison)
                        node_association_line[nodeIndex].set_data(X, Y)
                        node_association_line[nodeIndex].set_color('green')
                        node_association_line[nodeIndex].set_linestyle('--')
                        node_association_line[nodeIndex].set_linewidth(3)
                    else:
                        node_association_line[nodeIndex].set_data(X, Y)
                        node_association_line[nodeIndex].set_color(colorsBS[closestBSDownlink])
                        node_association_line[nodeIndex].set_linestyle('-')
                        node_association_line[nodeIndex].set_linewidth(0.5)
            else: # % Associate to a Macrocell
                X = [node_list[nodeIndex]["v_x"][timeIndex], BaseStations[closestBSDownlink, 0]]
                Y = [node_list[nodeIndex]["v_y"][timeIndex], BaseStations[closestBSDownlink, 1]]

                node_association_line[nodeIndex].set_data(X, Y)
                node_association_line[nodeIndex].set_color(colorsBS[closestBSDownlink])
                node_association_line[nodeIndex].set_linestyle('-')
                node_association_line[nodeIndex].set_linewidth(0.5)

                association_vector[0, nodeIndex] = closestBSDownlink # Associate.
                association_vector_overflow_alternative[0, nodeIndex] = 0                
                
                active_Cells[closestBSDownlink] = 1 # This cell does not count for the overall PoF power budget.
                baseStation_users[closestBSDownlink] += 1 # Add user.
        
        # Compute additional throughput parameters
        total_DL_Throughput = 0
        for nodeIndex in range(0, len(s_mobility['NB_NODES'])):
            SINRDLink = radio_utils.compute_sinr_dl([node_list[nodeIndex]["v_x"][timeIndex], node_list[nodeIndex]["v_y"][timeIndex]], BaseStations, association_vector[0][nodeIndex], alpha_loss, PMacroCells, PFemtoCells, NMacroCells, noise)
            naturalDL = 10**(SINRDLink/10)
            if association_vector[0][nodeIndex] <= NMacroCells:
                BW = MacroCellDownlinkBW
            else:
                BW = FemtoCellDownlinkBW
            RateDL = (BW/baseStation_users[int(association_vector[0][nodeIndex])]) * np.log2(1 + naturalDL)
            total_DL_Throughput += RateDL

        total_DL_Throughput_overflow_alternative = 0
        for nodeIndex in range(0, len(s_mobility['NB_NODES'])):
            if association_vector_overflow_alternative[0][nodeIndex] == 0.0:
                SINRDLink = radio_utils.compute_sinr_dl([node_list[nodeIndex]["v_x"][timeIndex], node_list[nodeIndex]["v_y"][timeIndex]], BaseStations, association_vector[0][nodeIndex], alpha_loss, PMacroCells, PFemtoCells, NMacroCells, noise)
                naturalDL = 10**(SINRDLink/10)
                if association_vector[0][nodeIndex] <= NMacroCells:
                    BW = MacroCellDownlinkBW
                    RateDL = (BW / (baseStation_users[int(association_vector[0][nodeIndex])] + np.sum(association_vector_overflow_alternative == association_vector_overflow_alternative[0][nodeIndex]))) * np.log2(1 + naturalDL)
                else:
                    BW = FemtoCellDownlinkBW
                    # Must '+' to avoid divide by zero, in MATLAB is '-'
                    RateDL = (BW/(baseStation_users[int(association_vector[0][nodeIndex])] + overflown_from[int(association_vector[0][nodeIndex])])) * np.log2(1+naturalDL)
                total_DL_Throughput_overflow_alternative += RateDL 
            else:
                SINRDLink = radio_utils.compute_sinr_dl([node_list[nodeIndex]["v_x"][timeIndex], node_list[nodeIndex]["v_y"][timeIndex]], BaseStations, association_vector_overflow_alternative[0][nodeIndex], alpha_loss, PMacroCells, PFemtoCells, NMacroCells, noise)
                naturalDL = 10**(SINRDLink/10)
                BW = MacroCellDownlinkBW
                RateDL = (BW/(baseStation_users[int(association_vector_overflow_alternative[0][nodeIndex])] + np.sum(association_vector_overflow_alternative[0] == association_vector_overflow_alternative[0][nodeIndex]))) * np.log2(1+naturalDL)
        total_DL_Throughput_overflow_alternative += RateDL

        # Throughput with ONLY Macrocells
        total_DL_Throughput_only_Macros = 0
        temporal_association_vector = np.zeros(NMacroCells, dtype=int)

        for nodeIndex in range(0, len(s_mobility['NB_NODES'])):
            cl = user_association_utils.search_closest_macro([node_list[nodeIndex]["v_x"][timeIndex], node_list[nodeIndex]["v_y"][timeIndex]], BaseStations[0:NMacroCells, 0:2])
            temporal_association_vector[cl] += 1
            SINRDLink = radio_utils.compute_sinr_dl([node_list[nodeIndex]["v_x"][timeIndex], node_list[nodeIndex]["v_y"][timeIndex]], BaseStations, cl, alpha_loss, PMacroCells, PFemtoCells, NMacroCells, noise)
            naturalDL = 10**(SINRDLink/10)
            BW = MacroCellDownlinkBW
            
            RateDL = (BW / temporal_association_vector[cl]) * np.log2(1 + naturalDL)
            total_DL_Throughput_only_Macros += RateDL

        # Compute the number of active Smallcells
        live_smallcell_occupancy[timeIndex] = np.sum(active_Cells[NMacroCells-1:-1])
        
        #CHECK
        #live_occupancy_plot.set_data(sim_times[:timeIndex], live_smallcell_occupancy)
        #max_occupancy_plot.set_data([0, sim_times[timeIndex]], [NFemtoCells, NFemtoCells])
        #used.set_text('Phantom Cells in ON state: {}'.format(live_smallcell_occupancy[timeIndex]))

        # Compute the total consumption
        live_smallcell_consumption[timeIndex] = (live_smallcell_occupancy[timeIndex] * small_cell_consumption_ON 
            + (NFemtoCells - live_smallcell_occupancy[timeIndex]) * small_cell_consumption_SLEEP)

        # Update system throughput
        live_throughput[timeIndex] = total_DL_Throughput
        live_throughput_NO_BATTERY[timeIndex] = total_DL_Throughput_overflow_alternative
        live_throughput_only_Macros[timeIndex] = total_DL_Throughput_only_Macros

        # Decide about battery recharging
        if live_smallcell_consumption[timeIndex] < max_energy_consumption:
            available = max_energy_consumption - live_smallcell_consumption[timeIndex]
            I = np.argmin(battery_vector[0])
            if battery_vector[0][I] < battery_capacity:
                charging_intensity = available / np.mean(small_cell_voltage_range)
                battery_vector[0][I] = min(battery_vector[0][I] + charging_intensity * (timeStep/3600), battery_capacity)
                if battery_state[I] == 0.0: battery_state[I] = 1.0
                elif battery_state[I] == 2.0: battery_state[I] = 3.0
        
        # Compute the number of active Smallcells
        live_smallcell_occupancy[timeIndex] = np.sum(active_Cells[NMacroCells-1:-1])

        battery_mean_values[timeIndex] = np.mean(battery_vector[0])

        #CHECK
        # PLOT THINGS!!
        # Update total consumption plot
        # live_consumption_plot.set_data(sim_times[0:timeIndex], live_smallcell_consumption)
        # max_consumption_plot.set_data([0, sim_times[timeIndex]], [small_cell_consumption_ON * NFemtoCells, small_cell_consumption_ON * NFemtoCells])
        # consuming_ON.set_text(f"Energy consumption (Active Femtocells): {live_smallcell_occupancy[timeIndex] * small_cell_consumption_ON} W")
        # consuming_SLEEP.set_text(f"Energy consumption (Idle Femtocells): {(NFemtoCells - live_smallcell_occupancy[timeIndex]) * small_cell_consumption_SLEEP} W")
        # consuming_TOTAL.set_text(f"Energy consumption (Total Femtocells): {live_smallcell_consumption[timeIndex]} W")
        #
        # # Update system throughput plot
        # live_throughput_plot.set_data(sim_times[timeIndex - (len(live_throughput) - (SMA_WINDOW - 1)) + 1:timeIndex], 
        #                             np.convolve(live_throughput / 10e6, np.ones((SMA_WINDOW,))/SMA_WINDOW, mode='valid'))
        # live_throughput_NO_BATTERY_plot.set_data(sim_times[timeIndex - (len(live_throughput_NO_BATTERY) - (SMA_WINDOW - 1)) + 1:timeIndex], 
        #                                         np.convolve(live_throughput_NO_BATTERY / 10e6, np.ones((SMA_WINDOW,))/SMA_WINDOW, mode='valid'))
        # live_throughput_only_Macros_plot.set_data(sim_times[timeIndex - (len(live_throughput_only_Macros) - (SMA_WINDOW - 1)) + 1:timeIndex], 
        #                                         np.convolve(live_throughput_only_Macros / 10e6, np.ones((SMA_WINDOW,))/SMA_WINDOW, mode='valid'))

        # Update battery states
        # for b in range(NFemtoCells):
        #     handleToThisBar[b].set_height(battery_vector[NMacroCells + b])
        #     handleToThisBar[b].set_facecolor(battery_color_codes[battery_state[0, NMacroCells+b]])
        
        plt.draw()
        plt.pause(0.1)
        print("Step ended. Plots updated!")
        
    # END
    print("Simulation complete!")
    
    ## Plotting output
    # 1
    fig, ax = plt.subplots()
    #ax.plot([0, sim_times], [NFemtoCells, NFemtoCells], 'r', label='Total Small cells')
    ax.axhline(y=NFemtoCells, color='r', label='Total Small cells')
    ax.plot(sim_times, live_smallcell_occupancy, 'g', label='Small cells being used')
    #ax.text(0, NFemtoCells - 1, f"Phantom Cells ON: {NFemtoCells - 1}")
    ax.legend()
    ax.set_title('Number of small cells under use')
    
    # 2
    fig, ax = plt.subplots()
    #ax.plot([0, sim_times], [small_cell_consumption_ON * NFemtoCells, small_cell_consumption_ON * NFemtoCells], 'r', label='Total always ON consumption [W]')
    ax.axhline(y=small_cell_consumption_ON * NFemtoCells, color='r', label='Total always ON consumption [W]')
    ax.plot(sim_times, live_smallcell_consumption, 'g', label='Live energy consumption [W]')
    #ax.text(1, small_cell_consumption_ON * NFemtoCells - 1, f"Energy consumption (Active Femtocells): {small_cell_consumption_ON * NFemtoCells - 1} W")
    #ax.text(1, small_cell_consumption_ON * NFemtoCells - 3, f"Energy consumption (Idle Femtocells): {small_cell_consumption_ON * NFemtoCells - 3} W")
    #ax.text(1, small_cell_consumption_ON * NFemtoCells - 5, f"Energy consumption (Total Femtocells): {small_cell_consumption_ON * NFemtoCells - 5} W")
    ax.legend()
    ax.set_title('Live energy consumption')
    
    # 3
    fig, ax = plt.subplots()
    ax.plot(sim_times, live_throughput/10e6, label='With battery system')
    ax.plot(sim_times, live_throughput_NO_BATTERY/10e6, 'r--', label='Without battery system')
    ax.plot(sim_times, live_throughput_only_Macros/10e6, 'g:.', label='Only Macrocells')
    ax.legend()
    ax.set_title('Live system throughput')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Throughput [Mb/s]')
    
    # 3, filtered one
    SMA_WINDOW = 5
    fig, ax = plt.subplots()
    X = sim_times[timeIndex-(len(live_throughput)-(SMA_WINDOW-1))+1:timeIndex]/3600
    Y = np.convolve(live_throughput/10e6, np.ones((SMA_WINDOW,))/SMA_WINDOW, mode='valid')
    ax.plot(X, Y[:-1], label='With battery system')
    
    X = sim_times[timeIndex-(len(live_throughput_NO_BATTERY)-(SMA_WINDOW-1))+1:timeIndex]/3600
    Y = np.convolve(live_throughput_NO_BATTERY/10e6, np.ones((SMA_WINDOW,))/SMA_WINDOW, mode='valid')
    ax.plot(X, Y[:-1], 'r--', label='Without battery system')
    
    X = sim_times[timeIndex-(len(live_throughput_only_Macros)-(SMA_WINDOW-1))+1:timeIndex]/3600 
    Y = np.convolve(live_throughput_only_Macros/10e6, np.ones((SMA_WINDOW,))/SMA_WINDOW, mode='valid')
    ax.plot(X, Y[:-1], 'g--o', label='Only Macrocells')
    
    ax.legend()
    ax.set_title('Live system throughput [Filtered]')
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Throughput (Mb/s)')
    
    # 4
    #fig, ax = plt.subplots()
    #for b in range(NFemtoCells):
    #    ax.bar(NMacroCells + b + 1, battery_vector[0, NMacroCells + b], color='b')
    #ax.set_title('Live battery state')
    
    # 5
    fig, ax = plt.subplots()
    ax.plot(sim_times, battery_mean_values, label='Battery mean capacity')
    ax.axhline(y=3.3, color='r',label="Max. voltage battery")
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Mean battery capacity [Ah]')
    ax.legend()
    
    plt.show(block=False)
    plt.pause(0.001)
    input("hit [enter] to end")
    plt.close('all')
    
if __name__ == '__main__':
    main()