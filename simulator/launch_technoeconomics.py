__author__ = "Gabriel Otero Perez (gaoterop@it.uc3m.es), Jose-Manuel Martinez-Caro (jmmartinez@e-lighthouse.com), Enrique Fernandez Sanchez (efernandez@e-lighthouse.com)"
__credits__ = ["Gabriel Otero Perez", "Jose-Manuel Martinez-Caro", "Enrique Fernandez Sanchez"]
__version__ = "1.1"
__maintainer__ = "Jose-Manuel Martinez-Caro"
__email__ = "jmmartinez@e-lighthouse.com"
__status__ = "Validated"

import numpy as np
import scipy.io, uuid, logging, sys

from simulator.bcolors import bcolors
import simulator.map_utils, simulator.mobility_utils, simulator.user_association_utils, simulator.radio_utils
import model.RegionsCalcs
from datetime import datetime, timezone

# Default input_parameters. Copy and modify ad-hoc  [Legacy version]
INPUT_PARAMETERS = {
        'Users': 30,
        'timeStep': 3600,                       # In seconds, 1 hour
        'Simulation_Time': 2592000,             # In seconds, 1 month of 30 days
        'NMacroCells': 20,
        'NFemtoCells': 134,
        'Maplimit': 40,                       # Size of Map grid, [dont touch]

        'battery_capacity': 3.3,                # Ah
        'small_cell_consumption_on': 0.7,       # In Watts
        'small_cell_consumption_sleep': 0.05,   # In Watts
        'small_cell_voltage_min': 0.028,        # In mVolts
        'small_cell_voltage_max': 0.033,        # In mVolts
        'mean_user_speed': 5.5,                 # In m/s
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
        'use_user_list': False,
        'save_output': False,
        'output_folder': None,
        'use_nice_setup_file': "mocks/pruebas_algoritmo/use_case_1.mat"
    }

logger = logging.getLogger(__name__)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL.PngImagePlugin").setLevel(logging.WARNING)

def execute_simulator(canvas_widget = None, progressbar_widget = None, run_name: str = "", input_parameters: dict = INPUT_PARAMETERS, config_parameters: dict = CONFIG_PARAMETERS, custom_parameters: dict = {}):
    if run_name == "":
        run_name = str(uuid.uuid4())[:8]

        
    
    now_utc = datetime.now(timezone.utc)
    now_str = now_utc.strftime('%Y%m%d_%H:%M')
    run_name = f"{now_str} - {run_name}"
    logger.info(f"Run_name: {run_name}")
    
    # if canvas_widget is None:
    #     # In CLI execution, Tk works better than Qt
    #     import matplotlib
    #     matplotlib.use('TkAgg')  # Set the Matplotlib backend
    
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
        
        max_energy_consumption_total = numberOfLasers * 1                               # 1 Watt each laser, total energy inside PoF Budget (no batteries related)
        max_energy_consumption_active = numberOfLasers * small_cell_consumption_ON      # One laser per femtocell
        
        min_user_speed = 1
        max_user_speed = 2 * input_parameters['mean_user_speed'] - min_user_speed    # Get the max value, [xmin, xmax] that satisfy the mean 
        
        battery_dict = {
            'small_cell_consumption_ON': small_cell_consumption_ON,
            'small_cell_consumption_SLEEP': small_cell_consumption_SLEEP,
            'small_cell_current_draw': small_cell_current_draw,
            'small_cell_voltage_range': small_cell_voltage_range,
            'max_energy_consumption_total': max_energy_consumption_total,
            'max_energy_consumption_active': max_energy_consumption_active,
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
        return
    
    if config_parameters['use_user_list']:
        logger.info("Using defined 'user_list', overriding Simulation Time to 50s...")
        Simulation_Time = 50
    
    if config_parameters['use_nice_setup']:
        # Use nice_setup from .mat file. Already selected distribution of BaseStations
        try:
            # TODO
            file_name = config_parameters.get('use_nice_setup_file', 'simulator/nice_setup.mat')
            nice_setup_mat = scipy.io.loadmat(file_name)
            BaseStations = nice_setup_mat['BaseStations']
            Stations = BaseStations.shape
            Npoints = Stations[0]
            logger.debug(f"Stations: {Stations}, NPoints: {Npoints}")

            NMacroCells = nice_setup_mat['NMacroCells'][0][0]
            NFemtoCells = nice_setup_mat['NFemtoCells'][0][0]
        except Exception as e:
            logger.error(bcolors.FAIL + 'Error importing the nice_setup.mat' + bcolors.ENDC)
            logger.error(e)
            return
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
            return


    try:
        # Setup Regions!

        # TODO: Modify here
        # Two Regions map, one for Macrocell coverage and another for Femtocell coveage
        macro_bs, femto_bs = BaseStations[:NMacroCells], BaseStations[NMacroCells:]
        
        # Femtocells
        Regions_fem, _ = model.RegionsCalcs.create_regions(
            np.array(femto_bs),
            alpha_loss,
            max_radius_km_list=[1] * len(femto_bs)
        )
        
        # Macrocells
        Regions_mac = simulator.map_utils.create_regions(
            Npoints=NMacroCells,
            BaseStations=macro_bs,
            alpha_loss=alpha_loss,
            config_parameters=config_parameters,
        )

        
        Regions = {}    # (BS Index, Region of BS)
        for ind, reg in Regions_fem.items(): # 
            Regions[ind + NMacroCells] = reg
        for ind, reg in Regions_mac.items():
            Regions[ind] = reg
                
        # Regions = simulator.map_utils.create_regions(Npoints=Npoints, 
                                                    #  BaseStations=BaseStations, 
                                                    #  ax=ax, 
                                                    #  alpha_loss=alpha_loss, 
                                                    #  config_parameters=config_parameters,
                                                    #  canvas_widget=canvas_widget)
   

   
        # TODO: Modify here
        # RegionsMacrocells
        basestation_dict = {
            'BaseStations': BaseStations,
            'Regions': Regions,
            # 'RegionsMacrocells': Regions_mac,
            'NMacroCells': NMacroCells,
            'NFemtoCells': NFemtoCells,
        }
    except Exception as e:
        logger.error(bcolors.FAIL + 'Error plotting the BSs coverage' + bcolors.ENDC)
        logger.error(e)
        sys.exit(0)  

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
        
        # TODO
        
        user_position = [user_list[userIndex]['v_x'][0], user_list[userIndex]['v_y'][0]]
        closestBSDownlink = simulator.map_utils.search_closest_bs_optimized(user_position, Regions, BaseStations, NMacroCells)
        
        # # If closestBs... == -1 -> no femto found so that covers the user, try with macrocell
        # if closestBSDownlink == -1:
        #     closestBSDownlink = simulator.map_utils.search_closest_bs(user_position, Regions_mac)
        # else:
        #     # If FemtoBS found: update its index, as the position of the closesBS 
        #     # only refers within the femtobs. Globally, on the BSList, the 
        #     # femtocells are in the range (NMacrocells:) (from NMacrocell to last)
        #     closestBSDownlink += NMacroCells

        active_Cells[closestBSDownlink] = 1

    user_dict = {
        'users': s_mobility["NB_USERS"],
        'user_list': user_list,
        'user_pos_plot': user_pos_plot,
        'user_association_line': user_association_line
    }


    # Start the simulation!
    
    logger.info("Using E-Lighthouse algorithm...")
    from simulator.algorithm_technoeconomics import PoF_simulation_ELighthouse_Technoeconomics
    
    eli = PoF_simulation_ELighthouse_Technoeconomics(sim_times=sim_times,
                                    basestation_data=basestation_dict,
                                    user_data=user_dict,
                                    battery_data=battery_dict,
                                    transmit_power_data=transmit_power_dict,
                                    elighthouse_parameters=custom_parameters)
    
    eli.start_simulation(sim_times=sim_times, timeStep=timeStep)
    
    eli.plot_output(sim_times=sim_times,
                        show_plots=config_parameters['show_plots'],
                        timeStep=timeStep,
                        is_gui=(canvas_widget is not None))

    if config_parameters['save_output']:
        eli.save_run(fig_map=None, 
                        sim_times=sim_times, 
                        run_name=run_name, 
                        output_folder=config_parameters['output_folder'])
        
    logger.info(f"Execution {run_name} finished!")