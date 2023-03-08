__author__ = "Gabriel Otero Perez (gaoterop@it.uc3m.es), Jose-Manuel Martinez-Caro (jmmartinez@e-lighthouse.com), Enrique Fernandez Sanchez (efernandez@e-lighthouse.com)"
__credits__ = ["Gabriel Otero Perez", "Jose-Manuel Martinez-Caro", "Enrique Fernandez Sanchez"]
__version__ = "1.1"
__maintainer__ = "Jose-Manuel Martinez-Caro"
__email__ = "jmmartinez@e-lighthouse.com"
__status__ = "Working on"

from shapely.geometry import Polygon, GeometryCollection, MultiPolygon
from matplotlib.patches import Polygon as MplPolygon
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io, os, uuid, logging, time

from simulator.bcolors import bcolors
import simulator.map_utils, simulator.mobility_utils, simulator.user_association_utils, simulator.radio_utils

# Default input_parameters. Copy and modify ad-hoc
INPUT_PARAMETERS = {
        'battery_capacity': 3.3,                # Ah
        'small_cell_consumption_on': 0.7,       # In Watts
        'small_cell_consumption_sleep': 0.05,   # In Watts
        'small_cell_voltage_min': 0.028,        # In Volts
        'small_cell_voltage_max': 0.033,        # In Volts
        'Maplimit': 1000,                       # Size of Map grid, [dont touch]
        'Users': 30,
        'max_user_speed': 10,                   # In m/s
        'Simulation_Time': 50,                  # In seconds
        'timeStep': 0.5,                        # In seconds
        'numberOfLasers': 5,
        'noise': 2.5e-14,
        'SMA_WINDOW': 5, 
        'TransmittingPower' : {
            'PMacroCells': 40,
            'PFemtoCells': 0.1,
            'PDevice': 0.1,
            'MacroCellDownlinkBW': 20e6,
            'FemtoCellDownlinkBW': 1e9,
            'alpha_loss': 4.0            
        }
    }

CONFIG_PARAMETERS = {
        'use_nice_setup': True,
        'use_node_list': False,
        'show_plots': True,
        'show_live_plots': False,
        'speed_live_plots': 0.05,
        'save_output': False
    }

logger = logging.getLogger(__name__)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL.PngImagePlugin").setLevel(logging.WARNING)

def execute_simulator(run_name: str = "", input_parameters: dict = INPUT_PARAMETERS, config_parameters: dict = CONFIG_PARAMETERS):
    if run_name == "":
        run_name = str(uuid.uuid4())[:8]
    logger.info(f"Run_name: {run_name}")
    
    # Import input_parameters from dict
    try:
        battery_capacity = input_parameters['battery_capacity']
        small_cell_consumption_ON = input_parameters['small_cell_consumption_on']
        small_cell_consumption_SLEEP = input_parameters['small_cell_consumption_sleep']
        Maplimit = input_parameters['Maplimit']
        Simulation_Time = input_parameters['Simulation_Time']
        Users = input_parameters['Users']
        timeStep = input_parameters['timeStep']
        numberOfLasers = input_parameters['numberOfLasers']
        noise = input_parameters['noise']
        SMA_WINDOW = input_parameters['SMA_WINDOW']
        small_cell_voltage_range = np.array([input_parameters['small_cell_voltage_min'], 
                                             input_parameters['small_cell_voltage_max']])
        
        PMacroCells = input_parameters['TransmittingPower']['PMacroCells']
        PFemtoCells = input_parameters['TransmittingPower']['PFemtoCells']
        alpha_loss = input_parameters['TransmittingPower']['alpha_loss']
        MacroCellDownlinkBW = input_parameters['TransmittingPower']['MacroCellDownlinkBW']
        FemtoCellDownlinkBW = input_parameters['TransmittingPower']['FemtoCellDownlinkBW']
        
        small_cell_current_draw = small_cell_consumption_ON/np.mean(small_cell_voltage_range)
        max_energy_consumption = numberOfLasers * small_cell_consumption_ON
    except Exception as e:
        logger.error(bcolors.FAIL + 'Error importing parameters into local variables' + bcolors.ENDC)
        logger.error(e)
    
    # Generate random distribution of BaseStations
    #try:
    #    NMacroCells = input_parameters['NMacroCells']
    #    NFemtoCells = input_parameters['NFemtoCells']
    #    WeightsTier1 = np.ones((1, NMacroCells))*PMacroCells
    #    WeightsTier2 = np.ones((1, NFemtoCells))*PFemtoCells
    #    BaseStations = np.zeros((NMacroCells + NFemtoCells, 3))
    #    # Settle Macro cells 
    #    BaseStations[0:NMacroCells,0] = Maplimit * np.random.uniform(size=NMacroCells, low=1, high=NMacroCells)
    #    BaseStations[0:NMacroCells,1] = Maplimit * np.random.uniform(size=NMacroCells, low=1, high=NMacroCells)
    #    BaseStations[0:NMacroCells,2] = WeightsTier1
    #    BaseStations[NMacroCells:,0] = Maplimit * np.random.uniform(size=NFemtoCells, low=1, high=NFemtoCells)
    #    BaseStations[NMacroCells:,1] = Maplimit * np.random.uniform(size=NFemtoCells, low=1, high=NFemtoCells)
    #    # print(BaseStations)
    #    Stations = BaseStations.shape
    #    Npoints = Stations[0] #actually here
    #except Exception as e:
    #    print(bcolors.FAIL + 'Error calculating intermediate variables' + bcolors.ENDC)
    #    print(e)

    # Use nice_setup from .mat file. Already selected distribution of BaseStations
    try:
        nice_setup_mat = scipy.io.loadmat('simulator/nice_setup.mat')
        BaseStations = nice_setup_mat['BaseStations']
        Stations = BaseStations.shape
        Npoints = Stations[0]
        logger.debug(f"Stations: {Stations}, NPoints: {Npoints}")
        
        NMacroCells = nice_setup_mat['NMacroCells'][0][0]
        NFemtoCells = nice_setup_mat['NFemtoCells'][0][0]
    except Exception as e:
        logger.error(bcolors.FAIL + 'Error importing the nice_setup.mat' + bcolors.ENDC)
        logger.error(e)

    try:
        colorsBS = np.zeros((Npoints, 3))
        fig_map, ax = plt.subplots()
        plt.axis([0, Maplimit, 0, Maplimit])
        for a in range(0,Npoints):
            colorsBS[a] = np.random.uniform(size=3, low=0, high=1)
            ax.plot(BaseStations[a,0], BaseStations[a,1], 'o',color = colorsBS[a])
            ax.text(BaseStations[a,0], BaseStations[a,1], 'P'+str(a) , ha='center', va='bottom')

        if config_parameters['show_plots']:
            plt.show(block=False)

    except Exception as e:
        logger.error(bcolors.FAIL + 'Error importing the printing the BSs' + bcolors.ENDC)
        logger.error(e)

    # Setup Regions!
    try:
        _WholeRegion = Polygon([(0,0), (0,1000), (1000,1000),(1000, 0), (0,0)])
        _UnsoldRegion = _WholeRegion
        Regions = {}
        
        for k in range(Npoints-1,-1,-1):
            logger.debug('-- k: ' + str(k))
            _Region = _UnsoldRegion
            for j in range(0,Npoints):
                if (j<k):

                    if(BaseStations[k,2] != BaseStations[j,2]):
                        _resp = simulator.map_utils.apollonius_circle_path_loss(BaseStations[k][:2], BaseStations[j][:2], BaseStations[k][2], BaseStations[j][2], alpha_loss)
                        _Circ = simulator.map_utils.get_circle(_resp)

                        _Reg2 = Polygon(_Circ)
                        if not _Reg2.is_valid:
                            _Reg2 = _Reg2.buffer(0)
                        _Region = _Region.intersection(_Reg2)
                    else:
                        _R = simulator.map_utils.get_dominance_area(BaseStations[k][:2], BaseStations[j][:2])
                        _Region = _Region.intersection(_R)

            Regions[k] = _Region
            
            if isinstance(_Region, GeometryCollection):
                for geom in _Region.geoms:
                    if isinstance(geom, Polygon):
                        _polygon = MplPolygon(geom.exterior.coords, facecolor=np.random.rand(3), alpha=0.5, edgecolor=None)
                        ax.add_patch(_polygon)
            elif isinstance(_Region, MultiPolygon):
                col = np.random.rand(3)
                logger.debug('MultiPolygon here!')
                for _Reg in _Region.geoms:
                    _polygon = MplPolygon(_Reg.exterior.coords, facecolor=col, alpha=0.5, edgecolor=None)
                    ax.add_patch(_polygon)

            else:
                _polygon = MplPolygon(_Region.exterior.coords, facecolor=np.random.rand(3), alpha=0.5, edgecolor=None)
                ax.add_patch(_polygon)

            _UnsoldRegion = _UnsoldRegion.difference(_Region)
            

            # Slow down for the viewer
            if config_parameters['show_plots']:
                plt.pause(config_parameters['speed_live_plots'])    
    except Exception as e:
        logger.error(bcolors.FAIL + 'Error plotting the BSs coverage' + bcolors.ENDC)
        logger.error(e)    

    sim_input = {
        'V_POSITION_X_INTERVAL': [0, Maplimit],                         # (m)
        'V_POSITION_Y_INTERVAL': [0, Maplimit],                         # (m)
        'V_SPEED_INTERVAL': [1, input_parameters['max_user_speed']],    # (m/s)
        'V_PAUSE_INTERVAL': [0, 3],                                     # pause time (s)
        'V_WALK_INTERVAL': [30.00, 60.00],                              # walk time(s)
        'V_DIRECTION_INTERVAL': [-180, 180],                            # (degrees)
        'SIMULATION_TIME': Simulation_Time,                             # (s)
        'NB_NODES': Users
    }
    logger.debug(sim_input['V_WALK_INTERVAL'])
    
    # Generate the mobility path of users
    s_mobility = simulator.mobility_utils.generate_mobility(sim_input)
    s_mobility["NB_NODES"] = []
    for node in range(0, sim_input['NB_NODES']):
        s_mobility['NB_NODES'].append(s_mobility[node])

    sim_times = np.arange(0, sim_input['SIMULATION_TIME'] + timeStep, timeStep)

    #  Create visualization plots
    node_list = []
    for nodeIndex in range(sim_input['NB_NODES']):
        node_y = np.interp(sim_times, s_mobility['V_TIME'][nodeIndex], s_mobility['V_POSITION_Y'][nodeIndex])
        node_x = np.interp(sim_times, s_mobility['V_TIME'][nodeIndex], s_mobility['V_POSITION_X'][nodeIndex])
        node_list.append({'v_x': node_x, 'v_y': node_y})

    ### Validate with MATLAB, import node_list with mobility data
    if config_parameters['use_node_list']:
        node_list_mat = scipy.io.loadmat('simulator/node_list.mat')
        node_list_mat = node_list_mat['node_list']
        for nodeIndex in range(0, sim_input['NB_NODES']):
            node_list[nodeIndex]['v_x'] = node_list_mat['v_x'][0][nodeIndex][0]
            node_list[nodeIndex]['v_y'] = node_list_mat['v_y'][0][nodeIndex][0]
    ###

    active_Cells = np.zeros(NMacroCells + NFemtoCells)
    node_pos_plot = []
    node_association_line = []

    for nodeIndex in range(sim_input['NB_NODES']):
        node_pos = ax.plot(node_list[nodeIndex]['v_x'][0], node_list[nodeIndex]['v_y'][0], '+', markersize=10, linewidth=2, color=[0.3, 0.3, 1])
        node_pos_plot.append(node_pos)

        closestBSDownlink = simulator.map_utils.search_closest_bs([node_list[nodeIndex]['v_x'][0], node_list[nodeIndex]['v_y'][0]], Regions)
        x = [node_list[nodeIndex]['v_x'][0], BaseStations[closestBSDownlink][0]]
        y = [node_list[nodeIndex]['v_y'][0], BaseStations[closestBSDownlink][1]]
        node_assoc, = ax.plot(x, y, color=colorsBS[closestBSDownlink])
        node_association_line.append(node_assoc)

        active_Cells[closestBSDownlink] = 1

    ax.set_title('Downlink association. Distance & Power criterion')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    text = ax.text(0, Maplimit, 'Time (sec) = 0')

    if config_parameters['show_plots']:
        plt.show(block=False)
    
    live_smallcell_occupancy = np.zeros(len(sim_times))
    live_smallcell_occupancy[0] = sum(active_Cells[NMacroCells-1:-1])
    
    live_smallcell_overflow = np.zeros(len(sim_times))
    live_smallcell_overflow[0] = sum(active_Cells[NMacroCells-1:-1])

    if config_parameters['show_live_plots']:
        fig, ax = plt.subplots()
        ax.plot([0, sim_times[0]], [NFemtoCells, NFemtoCells], 'r', label='Total Small cells')
        ax.plot(sim_times[0], live_smallcell_occupancy[0], 'g', label='Small cells being used')
        ax.text(0, NFemtoCells - 1, f"Phantom Cells ON: 0")
        ax.legend()
        ax.set_title('Number of small cells under use')

    # Plot the first time slot for consumption
    live_smallcell_consumption = np.zeros(len(sim_times))
    live_smallcell_consumption[0] = live_smallcell_occupancy[0] * small_cell_consumption_ON + (NFemtoCells - live_smallcell_occupancy[0]) * small_cell_consumption_SLEEP

    if config_parameters['show_live_plots']:
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

    if config_parameters['show_live_plots']:
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

    if config_parameters['show_live_plots']:
        fig, ax = plt.subplots()
        for b in range(NFemtoCells):
            ax.bar(NMacroCells + b + 1, battery_vector[0, NMacroCells + b], color='b')
        ax.set_title('Live battery state')
        
    if config_parameters['show_live_plots']:
        plt.show(block=False)

    # Start the simulation!
    logger.info("Starting simulation...")
    start = time.time()
    with tqdm(total=100, desc='Simulating...') as f:
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
            
                # Update position on plot of User/Node
                node_pos_plot[nodeIndex][0].set_data([node_list[nodeIndex]["v_x"][timeIndex], node_list[nodeIndex]["v_y"][timeIndex]])

                #Search serving base station
                closestBSDownlink = simulator.map_utils.search_closest_bs([node_list[nodeIndex]["v_x"][timeIndex], node_list[nodeIndex]["v_y"][timeIndex]], Regions)

                #If closest is a Femtocell and it is sleeping (it has no users), then, check total energy consumption
                if closestBSDownlink > NMacroCells:

                    if baseStation_users[closestBSDownlink] == 0: #If inactive

                        #Can I turn it on with PoF?
                        active_femto = np.sum(active_Cells[NMacroCells:])
                        current_watts = (active_femto * small_cell_consumption_ON) + ((NFemtoCells - active_femto) * small_cell_consumption_SLEEP)
                        if current_watts >= (max_energy_consumption - small_cell_consumption_ON + small_cell_consumption_SLEEP): # No, I cannot. Check battery.

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
                                association_vector_overflow_alternative[0, nodeIndex] = simulator.user_association_utils.search_closest_macro([node_list[nodeIndex]["v_x"][timeIndex], node_list[nodeIndex]["v_y"][timeIndex]], BaseStations[0:NMacroCells, 0:2])
                                overflown_from[closestBSDownlink] += 1

                                # Comment on MATLAB:
                                #active_Cells[closestBSDownlink] = 1 # This cell does not count for the overall PoF power budget.
                                battery_state[closestBSDownlink] = 2 # Discharge battery.
                                battery_vector[0, closestBSDownlink] = max(0, battery_vector[0, closestBSDownlink] - (timeStep/3600) * small_cell_current_draw) # However, draw from Femtocell's battery.
                                baseStation_users[closestBSDownlink] += 1 # Add user.
                            else:
                                #Associate to closest Macrocell
                                closest_Macro = simulator.user_association_utils.search_closest_macro([node_list[nodeIndex]["v_x"][timeIndex], node_list[nodeIndex]["v_y"][timeIndex]], BaseStations[0:NMacroCells, 0:2])
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
                        if battery_state[closestBSDownlink] == 2.0: # Is Discharging
                            # If we had no batteries, this user would have been gone to the closest macrocell. Search "overflow" alternative and add 1 to the "kicked" users of this femtocell in the hypothetical case we had no batteries installed. 
                            association_vector_overflow_alternative[0, nodeIndex] = simulator.user_association_utils.search_closest_macro([node_list[nodeIndex]["v_x"][timeIndex], node_list[nodeIndex]["v_y"][timeIndex]], BaseStations[0:NMacroCells, 0:2])
                            overflown_from[closestBSDownlink] += 1
                        else:
                            association_vector_overflow_alternative[0, nodeIndex] = 0
                        baseStation_users[closestBSDownlink] += 1 # Add user.

                        X = [node_list[nodeIndex]["v_x"][timeIndex], BaseStations[closestBSDownlink, 0]]
                        Y = [node_list[nodeIndex]["v_y"][timeIndex], BaseStations[closestBSDownlink, 1]]

                        if battery_state[closestBSDownlink] == 2.0: # Is Discharging
                            # %If using battery (only check == 2 because 3 only happens later at chaging decison)
                            node_association_line[nodeIndex].set_data(X, Y)
                            node_association_line[nodeIndex].set_color('green')
                            node_association_line[nodeIndex].set_linestyle('--')
                            node_association_line[nodeIndex].set_linewidth(3)
                        else:   # Is Charging
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

                    active_Cells[closestBSDownlink] = 1
                    baseStation_users[closestBSDownlink] += 1 # Add user.

            ## Compute additional throughput parameters
            # Throughput WITH batteries
            total_DL_Throughput = 0
            for nodeIndex in range(0, len(s_mobility['NB_NODES'])):
                SINRDLink = simulator.radio_utils.compute_sinr_dl([node_list[nodeIndex]["v_x"][timeIndex], node_list[nodeIndex]["v_y"][timeIndex]], BaseStations, association_vector[0][nodeIndex], alpha_loss, PMacroCells, PFemtoCells, NMacroCells, noise)
                naturalDL = 10**(SINRDLink/10)
                if association_vector[0][nodeIndex] < NMacroCells:
                    BW = MacroCellDownlinkBW
                else:
                    BW = FemtoCellDownlinkBW
                RateDL = (BW/baseStation_users[int(association_vector[0][nodeIndex])]) * np.log2(1 + naturalDL)
                total_DL_Throughput += RateDL

            # Throughput WITHOUT batteries
            total_DL_Throughput_overflow_alternative = 0
            for nodeIndex in range(0, len(s_mobility['NB_NODES'])):
                if association_vector_overflow_alternative[0][nodeIndex] == 0.0:
                    SINRDLink = simulator.radio_utils.compute_sinr_dl([node_list[nodeIndex]["v_x"][timeIndex], node_list[nodeIndex]["v_y"][timeIndex]], BaseStations, association_vector[0][nodeIndex], alpha_loss, PMacroCells, PFemtoCells, NMacroCells, noise)
                    naturalDL = 10**(SINRDLink/10)
                    if association_vector[0][nodeIndex] < NMacroCells:
                        BW = MacroCellDownlinkBW
                        RateDL = (BW / (baseStation_users[int(association_vector[0][nodeIndex])] + np.sum(association_vector_overflow_alternative == association_vector_overflow_alternative[0][nodeIndex]))) * np.log2(1 + naturalDL)
                    else:
                        BW = FemtoCellDownlinkBW
                        # Must '+' to avoid divide by zero, in MATLAB is '-'
                        RateDL = (BW/(baseStation_users[int(association_vector[0][nodeIndex])] + overflown_from[int(association_vector[0][nodeIndex])])) * np.log2(1+naturalDL)
                    total_DL_Throughput_overflow_alternative += RateDL 
                else:
                    SINRDLink = simulator.radio_utils.compute_sinr_dl([node_list[nodeIndex]["v_x"][timeIndex], node_list[nodeIndex]["v_y"][timeIndex]], BaseStations, association_vector_overflow_alternative[0][nodeIndex], alpha_loss, PMacroCells, PFemtoCells, NMacroCells, noise)
                    naturalDL = 10**(SINRDLink/10)
                    BW = MacroCellDownlinkBW
                    RateDL = (BW/(baseStation_users[int(association_vector_overflow_alternative[0][nodeIndex])] + np.sum(association_vector_overflow_alternative[0] == association_vector_overflow_alternative[0][nodeIndex]))) * np.log2(1+naturalDL)
                    total_DL_Throughput_overflow_alternative += RateDL

            # Throughput with ONLY Macrocells
            total_DL_Throughput_only_Macros = 0
            temporal_association_vector = np.zeros(NMacroCells, dtype=int)
            for nodeIndex in range(0, len(s_mobility['NB_NODES'])):
                cl = simulator.user_association_utils.search_closest_macro([node_list[nodeIndex]["v_x"][timeIndex], node_list[nodeIndex]["v_y"][timeIndex]], BaseStations[0:NMacroCells, 0:2])
                temporal_association_vector[cl] += 1
                SINRDLink = simulator.radio_utils.compute_sinr_dl([node_list[nodeIndex]["v_x"][timeIndex], node_list[nodeIndex]["v_y"][timeIndex]], BaseStations, cl, alpha_loss, PMacroCells, PFemtoCells, NMacroCells, noise)
                naturalDL = 10**(SINRDLink/10)
                BW = MacroCellDownlinkBW

                RateDL = (BW / temporal_association_vector[cl]) * np.log2(1 + naturalDL)
                total_DL_Throughput_only_Macros += RateDL

            # Compute the number of active Smallcells
            live_smallcell_occupancy[timeIndex] = np.count_nonzero(active_Cells[NMacroCells:])
            
            # Compute the cells that overflow
            live_smallcell_overflow[timeIndex] = np.count_nonzero(overflown_from[NMacroCells:])

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
                I = np.argmin(battery_vector[0])    # TODO: why only one battery decision per timeStep?
                if battery_vector[0][I] < battery_capacity:
                    charging_intensity = available / np.mean(small_cell_voltage_range)
                    battery_vector[0][I] = min(battery_vector[0][I] + charging_intensity * (timeStep/3600), battery_capacity)
                    if battery_state[I] == 0.0: battery_state[I] = 1.0      # If none state, set as charging
                    elif battery_state[I] == 2.0: battery_state[I] = 3.0    # If discharging, set as charging & discharging

            battery_mean_values[timeIndex] = np.mean(battery_vector[0])

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

            if config_parameters['show_plots']:
                plt.draw()
                plt.pause(config_parameters['speed_live_plots'])
                #print("Step ended. Plots updated!")
        
    # END
    logger.info("Simulation complete!")
    logger.info(f"Elapsed time: {np.round(time.time() - start, decimals=4)} seconds.")
    
    ## Plotting output
    # 1
    fig_cell_occupancy, ax = plt.subplots()
    ax.axhline(y=NFemtoCells, color='r', label='Total Small cells')
    ax.plot(sim_times, live_smallcell_occupancy, 'g', label='Small cells being used')
    ax.plot(sim_times, live_smallcell_overflow, 'b', label='Small cells overflowed')
    #ax.text(0, NFemtoCells - 1, f"Phantom Cells ON: {NFemtoCells - 1}")
    #ax.axhline(y=numberOfLasers-1, label=f"Max. Lasers:")
    ax.legend()
    ax.set_title('Number of small cells under use')
    
    # 2
    fig_cell_consumption, ax = plt.subplots()
    ax.axhline(y=small_cell_consumption_ON * NFemtoCells, color='r', label='Total always ON consumption [W]')
    ax.plot(sim_times, live_smallcell_consumption, 'g', label='Live energy consumption [W]')
    #ax.text(1, small_cell_consumption_ON * NFemtoCells - 1, f"Energy consumption (Active Femtocells): {small_cell_consumption_ON * NFemtoCells - 1} W")
    #ax.text(1, small_cell_consumption_ON * NFemtoCells - 3, f"Energy consumption (Idle Femtocells): {small_cell_consumption_ON * NFemtoCells - 3} W")
    #ax.text(1, small_cell_consumption_ON * NFemtoCells - 5, f"Energy consumption (Total Femtocells): {small_cell_consumption_ON * NFemtoCells - 5} W")
    ax.legend()
    ax.set_title('Live energy consumption')
    
    # 3
    fig_throughput, ax = plt.subplots()
    ax.plot(sim_times, live_throughput/10e6, label='With battery system')
    ax.plot(sim_times, live_throughput_NO_BATTERY/10e6, 'r--', label='Without battery system')
    ax.plot(sim_times, live_throughput_only_Macros/10e6, 'g:.', label='Only Macrocells')
    ax.legend()
    ax.set_title('Live system throughput')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Throughput [Mb/s]')
    
    # 3, filtered one
    SMA_WINDOW = input_parameters['SMA_WINDOW']
    fig_throughput_smooth, ax = plt.subplots()
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
    ax.set_title('Live system throughput [smooth]')
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Throughput (Mb/s)')
    
    # 4
    #fig, ax = plt.subplots()
    #for b in range(NFemtoCells):
    #    ax.bar(NMacroCells + b + 1, battery_vector[0, NMacroCells + b], color='b')
    #ax.set_title('Live battery state')
    
    # 5
    fig_battery_mean, ax = plt.subplots()
    ax.plot(sim_times, battery_mean_values, label='Battery mean capacity')
    ax.axhline(y=3.3, color='r',label="Max. voltage battery")
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Battery capacity [Ah]')
    ax.set_title('Mean Battery Capacity of the System')
    ax.legend()
    
    if config_parameters['show_plots']:
        plt.show(block=False)
        input("hit [enter] to close plots and continue")
        plt.close('all')
    
    if config_parameters['save_output']:
        logger.info("Saving output data...")
        # Save results to files as csv!
        root_folder = os.path.abspath(__file__ + os.sep + os.pardir + os.sep + os.pardir)
        output_folder = os.path.join(root_folder, 'output')
        os.makedirs(output_folder, exist_ok=True)
        logger.info(f"Output folder: {output_folder}")

        run_folder = os.path.join(output_folder, run_name)
        os.makedirs(run_folder, exist_ok=True)
        logger.info(f"Run folder: {run_folder}")

        # Create CSV and Plot folders
        csv_folder = os.path.join(run_folder, 'csv')
        plot_folder = os.path.join(run_folder, 'plot')
        os.makedirs(csv_folder, exist_ok=True)
        os.makedirs(plot_folder, exist_ok=True)

        # Save CSV foreach plot
        df_output = pd.DataFrame({'time[s]': sim_times,
                                  'small_cell_ocupacy': live_smallcell_occupancy,
                                  'small_cell_overflow': live_smallcell_overflow,
                                  'small_cell_consumption[W]': live_smallcell_consumption,
                                  'throughput[mbps]': live_throughput/10e6,
                                  'throughput_no_battery[mbps]': live_throughput_NO_BATTERY/10e6,
                                  'throughput_only_macro[mbps]': live_throughput_only_Macros/10e6,
                                  'battery_mean[Ah]': battery_mean_values})
        df_output = df_output.assign(NMacroCells=NMacroCells)
        df_output = df_output.assign(NFemtoCells=NFemtoCells)
        df_output.to_csv(os.path.join(csv_folder, f'{run_name}-output.csv'), index=False)
    
        # Save figures to output folder
        fig_map.savefig(os.path.join(plot_folder, f'{run_name}-map.png'))
        fig_cell_occupancy.savefig(os.path.join(plot_folder, f'{run_name}-cell_occupancy.png'))
        fig_cell_consumption.savefig(os.path.join(plot_folder, f'{run_name}-cell_consumption.png'))
        fig_throughput.savefig(os.path.join(plot_folder, f'{run_name}-throughput.png'))
        fig_throughput_smooth.savefig(os.path.join(plot_folder, f'{run_name}-throughput_smooth.png'))
        fig_battery_mean.savefig(os.path.join(plot_folder, f'{run_name}-battery_mean.png'))

        # Copy node_list.mat [replicability of run]
        node_list_output_mat_path = os.path.join(run_folder, f'{run_name}-node_list.mat')
        scipy.io.savemat(node_list_output_mat_path, {'node_list': node_list})
        
        # Copy nice_setup.mat [replicability of run]
        nice_setup_mat_path = os.path.join(run_folder, f'{run_name}-nice_setup.mat')
        nice_setup_struct = {
            'BaseStations': BaseStations,
            'NFemtoCells': NFemtoCells,
            'NMacroCells': NMacroCells,
        }
        scipy.io.savemat(nice_setup_mat_path, nice_setup_struct)

        logger.info("Succesfully saved output files")
    logger.info(f"Execution {run_name} finished!")