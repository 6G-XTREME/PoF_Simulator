__author__ = "Gabriel Otero Perez (gaoterop@it.uc3m.es), Jose-Manuel Martinez-Caro (jmmartinez@e-lighthouse.com), Enrique Fernandez Sanchez (efernandez@e-lighthouse.com)"
__credits__ = ["Gabriel Otero Perez", "Jose-Manuel Martinez-Caro", "Enrique Fernandez Sanchez"]
__version__ = "1.1"
__maintainer__ = "Jose-Manuel Martinez-Caro"
__email__ = "jmmartinez@e-lighthouse.com"
__status__ = "Validated"

import matplotlib.pyplot as plt
import numpy as np
import scipy.io, uuid, logging, sys
from simulator.bcolors import bcolors
import simulator.map_utils, simulator.mobility_utils, simulator.user_association_utils, simulator.radio_utils
import model.RegionsCalcs
from datetime import datetime, timezone


# Default input_parameters. Copy and modify ad-hoc  [Legacy version]
INPUT_PARAMETERS = {
        'Users': 1000,
        'timeStep': 3600,                       # In seconds, 1 hour
        'Simulation_Time': 7200,             # In seconds, Debug 2 steps
        # 'Simulation_Time': 2592000,             # In seconds, 1 month of 30 days
        # 'NMacroCells': 20,
        # 'NFemtoCells': 134,
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
        'use_nice_setup_file': "mocks/pruebas_algoritmo/use_case_1.mat",
        'show_plots': False,

        'use_user_list': False,
        'show_live_plots': False,
        'speed_live_plots': 0.001,
        'save_output': True,
        'output_folder': None,
    }


logger = logging.getLogger(__name__)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL.PngImagePlugin").setLevel(logging.WARNING)

def execute_simulator(canvas_widget = None, progressbar_widget = None, run_name: str = "", input_parameters: dict = INPUT_PARAMETERS, config_parameters: dict = CONFIG_PARAMETERS, custom_parameters: dict = {}):
    if run_name == "":
        run_name = str(uuid.uuid4())[:8]
    
    now_utc = datetime.now()
    now_str = now_utc.strftime('%Y%m%d_%H:%M')
    run_name = f"{now_str} - {run_name}"
    logger.info(f"Run_name: {run_name}")
    
    if canvas_widget is None:
        # In CLI execution, Tk works better than Qt
        import matplotlib
        # matplotlib.use('TkAgg')  # Set the Matplotlib backend
    
    # Import input_parameters from dict
    try:
        battery_capacity = input_parameters.get('battery_capacity', 3.3)
        small_cell_consumption_ON = input_parameters.get('small_cell_consumption_on', 0.7)
        small_cell_consumption_SLEEP = input_parameters.get('small_cell_consumption_sleep', 0.05)
        Maplimit = input_parameters.get('Maplimit', 40)
        Simulation_Time = input_parameters.get('Simulation_Time', 7200)
        Users = input_parameters.get('Users', 1000)
        timeStep = input_parameters.get('timeStep', 3600)
        numberOfLasers = input_parameters.get('numberOfLasers', 5)
        noise = input_parameters.get('noise', 2.5e-14)
        SMA_WINDOW = input_parameters.get('SMA_WINDOW', 5)
        small_cell_voltage_range = np.array([input_parameters.get('small_cell_voltage_min', 0.028), 
                                             input_parameters.get('small_cell_voltage_max', 0.033)])
        
        PMacroCells = input_parameters.get('TransmittingPower', {}).get('PMacroCells', 40)
        PFemtoCells = input_parameters.get('TransmittingPower', {}).get('PFemtoCells', 0.1)
        alpha_loss = input_parameters.get('TransmittingPower', {}).get('alpha_loss', 4.0)
        MacroCellDownlinkBW = input_parameters.get('TransmittingPower', {}).get('MacroCellDownlinkBW', 20e6)
        FemtoCellDownlinkBW = input_parameters.get('TransmittingPower', {}).get('FemtoCellDownlinkBW', 1e9)
        
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
    
    if config_parameters.get('use_user_list', False):
        logger.info("Using defined 'user_list', overriding Simulation Time to 50s...")
        Simulation_Time = 50
    
    if config_parameters.get('use_nice_setup', True):
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
            NMacroCells = input_parameters.get('NMacroCells')
            NFemtoCells = input_parameters.get('NFemtoCells')
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
        colorsBS = np.zeros((Npoints, 3))
        if canvas_widget is None: 
            fig_map, ax = plt.subplots()
            plt.axis([0, Maplimit, 0, Maplimit])
        else:
            canvas_widget.figure.clf()
            ax = canvas_widget.figure.add_subplot(111)
            fig_map = ax.figure
            ax.set_xlim(0, Maplimit)
            ax.set_ylim(0, Maplimit)
        for a in range(0,Npoints):
            colorsBS[a] = np.random.uniform(size=3, low=0, high=1)
            ax.plot(BaseStations[a,0], BaseStations[a,1], 'o',color = colorsBS[a])
            ax.text(BaseStations[a,0], BaseStations[a,1], 'P'+str(a) , ha='center', va='bottom')
        
        # Algorithm Centroid of Extra Point of Charge
        if 'extraPoFCharger' in custom_parameters:
            if custom_parameters.get('extraPoFCharger', False):
                if custom_parameters.get('typeExtraPoFCharger', "Random Macro") == "Random Macro":
                    # Select randomly a Macro Cell
                    selected_macro = np.random.randint(0, NMacroCells)
                    centroid_x = BaseStations[selected_macro, 0]
                    centroid_y = BaseStations[selected_macro, 1]  
                    custom_parameters['selected_random_macro'] = selected_macro
                    logger.info(f"Using random macro method, selected macro: {selected_macro}")
                else:   # Use "Centroid"
                    centroid_x = np.mean(BaseStations[NMacroCells:, 0])
                    centroid_y = np.mean(BaseStations[NMacroCells:, 1])
                custom_parameters['centroid_x'] = centroid_x
                custom_parameters['centroid_y'] = centroid_y
                ax.plot(centroid_x, centroid_y, 'x', color='red', markersize=10, markeredgewidth= 2, label='Centroid')
                #ax.text(centroid_x, centroid_y, 'Centroid', ha='center', va='bottom')
        
        if config_parameters.get('show_plots', False):
            if canvas_widget is None: 
                plt.show(block=False)  
            else: 
                canvas_widget.draw() 

    except Exception as e:
        logger.error(bcolors.FAIL + 'Error importing the printing the BSs' + bcolors.ENDC)
        logger.error(e)
        return

    try:
        # TODO: Modify here
        
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
        
        # Setup Regions!
        # Regions = simulator.map_utils.create_regions(Npoints=Npoints, 
                                                    #  BaseStations=BaseStations, 
                                                    #  ax=ax, 
                                                    #  alpha_loss=alpha_loss, 
                                                    #  config_parameters=config_parameters,
                                                    #  canvas_widget=canvas_widget)
   
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
    if config_parameters.get('use_user_list', False):
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
        # TODO: modify here
        user_position = [user_list[userIndex]['v_x'][0], user_list[userIndex]['v_y'][0]]
        closestBSDownlink = simulator.map_utils.search_closest_bs_optimized(user_position, Regions, BaseStations, NMacroCells)

        user_pos = ax.plot(user_list[userIndex]['v_x'][0], user_list[userIndex]['v_y'][0], '+', markersize=10, linewidth=2, color=[0.3, 0.3, 1])
        user_pos_plot.append(user_pos)

        # closestBSDownlink = simulator.map_utils.search_closest_bs([user_list[userIndex]['v_x'][0], user_list[userIndex]['v_y'][0]], Regions)
        x = [user_list[userIndex]['v_x'][0], BaseStations[closestBSDownlink][0]]
        y = [user_list[userIndex]['v_y'][0], BaseStations[closestBSDownlink][1]]
        user_assoc, = ax.plot(x, y, color=colorsBS[closestBSDownlink])
        user_association_line.append(user_assoc)

        active_Cells[closestBSDownlink] = 1

    ax.set_title('Downlink association. Distance & Power criterion')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    
    if canvas_widget is None:   # Only show time in the plot when is outside de UI
        text_plot = ax.text(0, Maplimit, 'Time (sec) = 0')
    else:
        text_plot = None

    user_dict = {
        'users': s_mobility["NB_USERS"],
        'user_list': user_list,
        'user_pos_plot': user_pos_plot,
        'user_association_line': user_association_line
    }

    if config_parameters.get('show_plots', False):
        if canvas_widget is None :
            plt.show(block=False)
        else:
            canvas_widget.draw()

    # Start the simulation!
    logger.info("Using E-Lighthouse algorithm...")
    from simulator.algorithm_tecno_new import PoF_simulation_ELighthouse_TecnoAnalysis
    
    eli = PoF_simulation_ELighthouse_TecnoAnalysis(sim_times=sim_times,
                                    basestation_data=basestation_dict,
                                    user_data=user_dict,
                                    battery_data=battery_dict,
                                    transmit_power_data=transmit_power_dict,
                                    elighthouse_parameters=custom_parameters)
    
    eli.start_simulation(sim_times=sim_times, 
                            timeStep=timeStep,
                            text_plot=text_plot,
                            show_plots=config_parameters.get('show_plots', False),
                            speed_plot=config_parameters.get('speed_live_plots', 0.001),
                            canvas_widget=canvas_widget,
                            progressbar_widget=progressbar_widget)
    
    eli.plot_output(sim_times=sim_times,
                        show_plots=config_parameters.get('show_plots', False),
                        timeStep=timeStep,
                        is_gui=(canvas_widget is not None))

    if config_parameters.get('save_output', True):
        eli.save_run(fig_map=fig_map, 
                        sim_times=sim_times, 
                        run_name=run_name, 
                        output_folder=config_parameters.get('output_folder', None))
        
    logger.info(f"Execution {run_name} finished!")