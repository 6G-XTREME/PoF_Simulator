import logging

#from simulator.launch import INPUT_PARAMETERS, CONFIG_PARAMETERS
from simulator.launch import execute_simulator

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
        'algorithm': 'uc3m',         # Select over: uc3m or elighthouse
        'use_nice_setup': True,
        'use_user_list': False,
        'show_plots': True,
        'show_live_plots': False,
        'speed_live_plots': 0.05,
        'save_output': False,
        'output_folder': None,
    }

logging.basicConfig(format='%(asctime)s %(levelname)s:%(module)s:%(funcName)s: %(message)s', 
                    level=logging.INFO)     # Change level to DEBUG if needed

if __name__ == "__main__":
    logging.info("Configuring input_parameters & config_simulator...")
    simulator_input = INPUT_PARAMETERS.copy()   # Retrieve the default input parameters
    simulator_input['Simulation_Time'] = 30*60  
    simulator_input['Users'] = 180
    
    simulator_config = CONFIG_PARAMETERS.copy() # Retrieve the default config of the simulator
    #simulator_config['use_user_list'] = True    # For validate MATLAB output. Always the same execution. Fixed Simulation Time
    simulator_config['speed_live_plots'] = 0.001
    simulator_config['save_output'] = True      # Save the run output on output folder
    simulator_config['show_plots'] = True       # Show output plots
    simulator_config['use_nice_setup'] = True
    #simulator_config['output_folder'] = 'simulation_test'
    simulator_config['algorithm'] = "elighthouse"
    
    # Custom Parameters E-Lighthouse
    custom_config = {}
    custom_config['user_report_position'] = 8   # For each four timeSteps, the users updates position
    custom_config['startup_max_tokens'] = 8     # TimeSlots to startup a FemtoCell
    custom_config['poweroff_unused_cell'] = 8   # TimeSlots to poweroff an unused Cell
    
    logging.info("Execute simulator...")
    #execute_simulator(config_parameters=simulator_config, custom_parameters=custom_config)
    
    ## Other ways to execute a simulator
    # Changing the simulator input
    execute_simulator(input_parameters=simulator_input, config_parameters=simulator_config, custom_parameters=custom_config)
    # Setting up the run_name
    #execute_simulator(run_name="my_run", input_parameters=simulator_input, config_parameters=simulator_config)