import logging
from simulator.launch_tecno_new import execute_simulator

# Default input_parameters. Copy and modify ad-hoc  [Legacy version]
INPUT_PARAMETERS = {
        'Users': 1000,
        'timeStep': 3600,                       # In seconds, 1 hour
        # 'Simulation_Time': 72000,             # In seconds, Debug 2 steps
        'Simulation_Time': 720000,             # In seconds, Debug 2 steps
        # 'Simulation_Time': 2592000,             # In seconds, 1 month of 30 days
        # 'NMacroCells': 20,
        # 'NFemtoCells': 134,
        'Maplimit': 40,                       # Size of Map grid, [dont touch]
        'numberOfLasers': 20,                   # Manually changed? Def 5. Should I input the topology instead? 
        
        'battery_capacity': 3.3,                # Ah
        'small_cell_consumption_on': 0.7,       # In Watts
        'small_cell_consumption_sleep': 0.05,   # In Watts
        'small_cell_voltage_min': 0.028,        # In mVolts
        'small_cell_voltage_max': 0.033,        # In mVolts
        'mean_user_speed': 5.5,                 # In m/s
        'noise': 2.5e-14,
        'SMA_WINDOW': 1,
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

CUSTOM_CONFIG = {
    'user_report_position': 1,  # For each four timeSteps, the users updates position
    'startup_max_tokens': 0,   # TimeSlots to startup a FemtoCell
    'poweroff_unused_cell': 0, # TimeSlots to poweroff an unused Cell
    'extraPoFCharger': False,     # Enable an extra Charger with 1W on the centroid
    'typeExtraPoFCharger': "Centroid",
    'use_harvesting': False,      # Enable the Solar Harvesting Mode -> New graph + solar charging...
    'weather': "RAINY",          # Select over SUNNY, CLOUDY or RAINY
    'city': "Cartagena",         # Select city
    'MapScale': 1,               # 1 km == 1 points (1:1)
    'fiberAttdBperKm': 0.2,      # Fiber attenuation in dB/Km
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