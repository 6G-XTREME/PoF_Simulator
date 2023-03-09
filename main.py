import logging

from simulator.launch import INPUT_PARAMETERS, CONFIG_PARAMETERS
from simulator.launch import execute_simulator

logging.basicConfig(format='%(asctime)s %(levelname)s:%(module)s:%(funcName)s: %(message)s', 
                    level=logging.INFO)     # Change level to DEBUG if needed

if __name__ == "__main__":
    logging.info("Configuring input_parameters & config_simulator...")
    simulator_input = INPUT_PARAMETERS.copy()   # Retrieve the default input parameters
    #simulator_input['Simulation_Time'] = 2000
    
    simulator_config = CONFIG_PARAMETERS.copy() # Retrieve the default config of the simulator
    simulator_config['use_node_list'] = True    # For validate MATLAB output. Always the same execution. Fixed Simulation Time
    simulator_config['save_output'] = False      # Save the run output on output folder
    simulator_config['show_plots'] = True       # Show output plots
    simulator_config['use_nice_setup'] = True
    
    logging.info("Execute simulator...")
    execute_simulator(config_parameters=simulator_config)
    
    # Other ways to execute a simulator
    # Changing the simulator input
    #execute_simulator(input_parameters=simulator_input, config_parameters=simulator_config)
    # Setting up the run_name
    #execute_simulator(run_name="my_run", input_parameters=simulator_input, config_parameters=simulator_config)