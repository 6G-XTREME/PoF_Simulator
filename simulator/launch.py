__author__ = "Gabriel Otero Perez (gaoterop@it.uc3m.es), Jose-Manuel Martinez-Caro (jmmartinez@e-lighthouse.com), Enrique Fernandez Sanchez (efernandez@e-lighthouse.com)"
__credits__ = ["Gabriel Otero Perez", "Jose-Manuel Martinez-Caro", "Enrique Fernandez Sanchez"]
__version__ = "1.1"
__maintainer__ = "Jose-Manuel Martinez-Caro"
__email__ = "jmmartinez@e-lighthouse.com"
__status__ = "Working on"

from shapely.geometry import Polygon, GeometryCollection, MultiPolygon
from matplotlib.patches import Polygon as MplPolygon
import matplotlib.pyplot as plt
import numpy as np
import scipy.io, uuid, logging

from simulator.bcolors import bcolors
import simulator.map_utils, simulator.mobility_utils, simulator.user_association_utils, simulator.radio_utils

# Default input_parameters. Copy and modify ad-hoc  [Legacy version]
INPUT_PARAMETERS = {
        'battery_capacity': 3.3,                # Ah
        'small_cell_consumption_on': 0.7,       # In Watts
        'small_cell_consumption_sleep': 0.05,   # In Watts
        'small_cell_voltage_min': 0.028,        # In mVolts
        'small_cell_voltage_max': 0.033,        # In mVolts
        'Maplimit': 1000,                       # Size of Map grid, [dont touch]
        'Users': 30,
        'mean_user_speed': 5.5,                 # In m/s
        'Simulation_Time': 50,                  # In seconds
        'timeStep': 0.5,                        # In seconds
        'numberOfLasers': 5,
        'noise': 2.5e-14,
        'SMA_WINDOW': 5, 
        'NMacroCells': 3,
        'NFemtoCells': 20,
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
        'algorithm': 'uc3m',         # Select over: uc3m or e-li
        'use_nice_setup': True,
        'use_user_list': False,
        'show_plots': True,
        'show_live_plots': False,
        'speed_live_plots': 0.05,
        'save_output': False,
        'output_folder': None,
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
        
        min_user_speed = 1
        max_user_speed = 2 * input_parameters['mean_user_speed'] - min_user_speed    # Get the max value, [xmin, xmax] that satisfy the mean 
        
        battery_dict = {
            'small_cell_consumption_ON': small_cell_consumption_ON,
            'small_cell_consumption_SLEEP': small_cell_consumption_SLEEP,
            'small_cell_current_draw': small_cell_current_draw,
            'small_cell_voltage_range': small_cell_voltage_range,
            'max_energy_consumption': max_energy_consumption,
            'battery_capacity': battery_capacity
        }
        
        transmit_power_dict = {
            'noise': noise,
            'alpha_loss': alpha_loss,
            'PMacroCells': PMacroCells,
            'PFemtoCells': PFemtoCells,
            'MacroCellDownlinkBW': MacroCellDownlinkBW,
            'FemtoCellDownlinkBW': FemtoCellDownlinkBW
        }
    except Exception as e:
        logger.error(bcolors.FAIL + 'Error importing parameters into local variables' + bcolors.ENDC)
        logger.error(e)
    
    if config_parameters['use_user_list']:
        logger.info("Using defined 'user_list', overriding Simulation Time to 50s...")
        Simulation_Time = 50
    
    if config_parameters['use_nice_setup']:
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
    else:   
        # Generate random distribution of BaseStations
        try:
            NMacroCells = input_parameters['NMacroCells']
            NFemtoCells = input_parameters['NFemtoCells']
            WeightsTier1 = np.ones((1, NMacroCells))*PMacroCells
            WeightsTier2 = np.ones((1, NFemtoCells))*PFemtoCells
            BaseStations = np.zeros((NMacroCells + NFemtoCells, 3))
            
            # Settle Macro cells 
            BaseStations[0:NMacroCells,0] = Maplimit * np.random.rand(NMacroCells)
            BaseStations[0:NMacroCells,1] = Maplimit * np.random.rand(NMacroCells)
            BaseStations[0:NMacroCells,2] = WeightsTier1
            
            # Settle Femto cells
            BaseStations[NMacroCells:,0] = Maplimit * np.random.rand(NFemtoCells)
            BaseStations[NMacroCells:,1] = Maplimit * np.random.rand(NFemtoCells)
            BaseStations[NMacroCells:,2] = WeightsTier2
            
            Stations = BaseStations.shape
            Npoints = Stations[0]
            logger.debug(f"Stations: {Stations}, NPoints: {Npoints}")
        except Exception as e:
            logger.error(bcolors.FAIL + 'Error calculating intermediate variables' + bcolors.ENDC)
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
   
        basestation_dict = {
            'BaseStations': BaseStations,
            'Regions': Regions,
            'NMacroCells': NMacroCells,
            'NFemtoCells': NFemtoCells,
            'colorsBS': colorsBS
        }
    except Exception as e:
        logger.error(bcolors.FAIL + 'Error plotting the BSs coverage' + bcolors.ENDC)
        logger.error(e)    

    sim_input = {
        'V_POSITION_X_INTERVAL': [0, Maplimit],                         # (m)
        'V_POSITION_Y_INTERVAL': [0, Maplimit],                         # (m)
        'V_SPEED_INTERVAL': [min_user_speed, max_user_speed],           # (m/s)
        'V_PAUSE_INTERVAL': [0, 3],                                     # pause time (s)
        'V_WALK_INTERVAL': [30.00, 60.00],                              # walk time(s)
        'V_DIRECTION_INTERVAL': [-180, 180],                            # (degrees)
        'SIMULATION_TIME': Simulation_Time,                             # (s)
        'NB_USERS': Users
    }
    logger.debug(sim_input['V_WALK_INTERVAL'])
    
    # Generate the mobility path of users
    s_mobility = simulator.mobility_utils.generate_mobility(sim_input)
    s_mobility["NB_USERS"] = []
    for user in range(0, sim_input['NB_USERS']):
        s_mobility['NB_USERS'].append(s_mobility[user])

    sim_times = np.arange(0, sim_input['SIMULATION_TIME'] + timeStep, timeStep)

    #  Create visualization plots
    user_list = []
    for userIndex in range(sim_input['NB_USERS']):
        user_y = np.interp(sim_times, s_mobility['V_TIME'][userIndex], s_mobility['V_POSITION_Y'][userIndex])
        user_x = np.interp(sim_times, s_mobility['V_TIME'][userIndex], s_mobility['V_POSITION_X'][userIndex])
        user_list.append({'v_x': user_x, 'v_y': user_y})

    ### Validate with MATLAB, import user_list with mobility data
    if config_parameters['use_user_list']:
        user_list_mat = scipy.io.loadmat('simulator/user_list.mat')
        try:
            user_list_mat = user_list_mat['user_list']
        except KeyError:
            user_list_mat = user_list_mat['node_list']
            
        for userIndex in range(0, sim_input['NB_USERS']):
            user_list[userIndex]['v_x'] = user_list_mat['v_x'][0][userIndex][0]
            user_list[userIndex]['v_y'] = user_list_mat['v_y'][0][userIndex][0]
    ###

    active_Cells = np.zeros(NMacroCells + NFemtoCells)
    user_pos_plot = []
    user_association_line = []

    for userIndex in range(sim_input['NB_USERS']):
        user_pos = ax.plot(user_list[userIndex]['v_x'][0], user_list[userIndex]['v_y'][0], '+', markersize=10, linewidth=2, color=[0.3, 0.3, 1])
        user_pos_plot.append(user_pos)

        closestBSDownlink = simulator.map_utils.search_closest_bs([user_list[userIndex]['v_x'][0], user_list[userIndex]['v_y'][0]], Regions)
        x = [user_list[userIndex]['v_x'][0], BaseStations[closestBSDownlink][0]]
        y = [user_list[userIndex]['v_y'][0], BaseStations[closestBSDownlink][1]]
        user_assoc, = ax.plot(x, y, color=colorsBS[closestBSDownlink])
        user_association_line.append(user_assoc)

        active_Cells[closestBSDownlink] = 1

    ax.set_title('Downlink association. Distance & Power criterion')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    text_plot = ax.text(0, Maplimit, 'Time (sec) = 0')

    user_dict = {
        'number_users': s_mobility["NB_USERS"],
        'user_list': user_list,
        'user_pos_plot': user_pos_plot,
        'user_association_line': user_association_line
    }

    if config_parameters['show_plots']:
        plt.show(block=False)

    # Start the simulation!
    if config_parameters['algorithm'].lower() == 'uc3m':
        logger.info("Using UC3M algorithm...")
        from simulator.algorithm_uc3m import PoF_simulation_UC3M
        uc3m = PoF_simulation_UC3M(sim_times=sim_times,
                                   basestation_data=basestation_dict,
                                   user_data=user_dict,
                                   battery_data=battery_dict,
                                   transmit_power_data=transmit_power_dict)

        uc3m.start_simulation(sim_times=sim_times, 
                              timeStep=timeStep,
                              text_plot=text_plot,
                              show_plots=config_parameters['show_plots'],
                              speed_plot=config_parameters['speed_live_plots'])

        uc3m.plot_output(sim_times=sim_times,
                         show_plots=config_parameters['show_plots'])

        if config_parameters['save_output']:
            uc3m.save_run(fig_map=fig_map, 
                          sim_times=sim_times, 
                          run_name=run_name, 
                          output_folder=config_parameters['output_folder'])
    elif config_parameters['algorithm'].lower() == 'eli':
        logger.info("Using E-Lighthouse algorithm...")
        from simulator.algorithm_eli import PoF_simulation_ELi
        eli = PoF_simulation_ELi(sim_times=sim_times,
                                basestation_data=basestation_dict,
                                user_data=user_dict,
                                battery_data=battery_dict,
                                transmit_power_data=transmit_power_dict)
        
        eli.start_simulation(sim_times=sim_times, 
                             timeStep=timeStep,
                             text_plot=text_plot,
                             show_plots=config_parameters['show_plots'],
                             speed_plot=config_parameters['speed_live_plots'])
        
        eli.plot_output(sim_times=sim_times,
                         show_plots=config_parameters['show_plots'])

        if config_parameters['save_output']:
            eli.save_run(fig_map=fig_map, 
                          sim_times=sim_times, 
                          run_name=run_name, 
                          output_folder=config_parameters['output_folder'])
    else:
        logger.error(f"Unable to select algorithm {config_parameters['algorithm']}")
        
    logger.info(f"Execution {run_name} finished!")