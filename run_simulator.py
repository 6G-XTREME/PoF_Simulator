import logging

from simulator.launch import INPUT_PARAMETERS, CONFIG_PARAMETERS
from simulator.launch import execute_simulator

logging.basicConfig(format='%(asctime)s %(levelname)s:%(module)s:%(funcName)s: %(message)s', 
                    level=logging.INFO)     # Change level to DEBUG if needed

if __name__ == "__main__":
    logging.info("Configuring input_parameters & config_simulator...")
    simulator_input = INPUT_PARAMETERS.copy()   # Retrieve the default input parameters
    simulator_input['Simulation_Time'] = 1*60  
    simulator_input['Users'] = 90
    
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
    
    # Other ways to execute a simulator
    # Changing the simulator input
    execute_simulator(input_parameters=simulator_input, config_parameters=simulator_config, custom_parameters=custom_config)
    # Setting up the run_name
    #execute_simulator(run_name="my_run", input_parameters=simulator_input, config_parameters=simulator_config)