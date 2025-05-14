import logging
from simulator.launch_tecno_new import execute_simulator

# Default input_parameters. Copy and modify ad-hoc  [Legacy version]
INPUT_PARAMETERS = {
        'Users': 1000,
        'UserMobilityType': "RANDOM",           # STATIC (random initial positions, same positions all the time steps)
                                                # RANDOM (random initial positions, random positions each time step)
                                                # MOBILE (random walk)
        'timeStep': 60*8,                       # In seconds, 5 minutes
        'Simulation_Time': 3600*24*1,             # In seconds, 24 hours
        # 'NMacroCells': 20,
        # 'NFemtoCells': 134,
        'Maplimit': 40,                         # Size of Map grid, [dont touch]
        'numberOfPofPools': 20,                         # Number of HPLDS
        'numberOfLasersPerPool': 5,                     # Number of lasers per HPLD
        'wattsPerLaser': 1,                     # Watts. Capacity of each HPLD

        'battery_capacity': 3.3,                # Ah
        'small_cell_consumption_on': 0.7,       # In Watts
        'small_cell_consumption_sleep': 0.05,   # In Watts
        'small_cell_voltage_min': 0.028,        # In mVolts
        'small_cell_voltage_max': 0.033,        # In mVolts
        'mean_user_speed': 5.5,                 # In m/s
        'noise': 2.5e-14,
        'SMA_WINDOW': 5, 
        'TransmittingPower' : {
            'PMacroCells': 40,
            'PFemtoCells': 0.1,
            'PDevice': 0.1,
            'MacroCellDownlinkBW': 20e6,
            'FemtoCellDownlinkBW': 1e9,
            'alpha_loss': 4.0            
        },
        'simultaneous_charging_batteries': "6", # ALL, x%, NUM
        'charging_battery_threshold': 0.95,     # (0, 1) %
    }

CONFIG_PARAMETERS = {
        'use_nice_setup': True,
        'use_nice_setup_file': "mocks/pruebas_algoritmo/UC1-S3-AllHL4withHPLD-AllHL5withFemto.mat",
        # 'use_nice_setup_file': "mocks/pruebas_algoritmo/use_case_1.mat",
        'show_plots': False,

        'use_user_list': False,
        'show_live_plots': False,
        'speed_live_plots': 0.001,
        'save_output': True,
        'output_folder': None,
        
        # Figure configuration
        'figure_config': {
            'fig_size': (12, 8),  # Default figure size (width, height) in inches
            'dpi': 200,           # Default DPI for saved figures
            'line_width': 1.5,    # Default line width for plots
            'font_size': 12,      # Default font size
            'tick_size': 10,      # Default tick label size
        },
    }

CUSTOM_CONFIG = {
    'user_report_position': 1,  # For each four timeSteps, the users updates position
    'startup_max_tokens': 1,   # TimeSlots to startup a FemtoCell
    'poweroff_unused_cell': 1, # TimeSlots to poweroff an unused Cell
    'extraPoFCharger': False,     # Enable an extra Charger with 1W on the centroid
    'typeExtraPoFCharger': "Centroid",
    'use_harvesting': False,      # Enable the Solar Harvesting Mode -> New graph + solar charging...
    'weather': "RAINY",          # Select over SUNNY, CLOUDY or RAINY
    'city': "Cartagena",         # Select city
    'MapScale': 1,               # How many km are 1 point in the source topology
    'fiberAttdBperKm': 0.2,      # Fiber attenuation in dB/Km
    'plot_dpi': 200,
    'config_times': {
        'femto_boot_time_seconds': 30,          # Time to boot a femto cell
        'femto_shutdown_time_seconds': 30,      # Time to shutdown a femto cell
        'time_to_shutdown_unused_femto': 60,    # Time to shutdown an unused femto cell
        
    },
    'seed': 1234567890,
}

logging.basicConfig(format='%(asctime)s %(levelname)s:%(module)s:%(funcName)s: %(message)s', 
                    level=logging.INFO)     # Change level to DEBUG if needed

if __name__ == "__main__":
    logging.info("Configuring input_parameters & config_simulator...")
    simulator_input = INPUT_PARAMETERS.copy()   # Retrieve the default input parameters
    simulator_config = CONFIG_PARAMETERS.copy() # Retrieve the default config of the simulator
    custom_config = CUSTOM_CONFIG.copy()
    
    logging.info("Execute simulator...")
    execute_simulator(input_parameters=simulator_input, config_parameters=simulator_config, custom_parameters=custom_config)