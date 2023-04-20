import logging, time, os
import numpy as np
import pandas as pd

from simulator.launch import INPUT_PARAMETERS, CONFIG_PARAMETERS
from simulator.launch import execute_simulator

BATCHS = 5
BATCH_NAME = 'batch-2'

logging.basicConfig(format='%(asctime)s %(levelname)s:%(module)s:%(funcName)s: %(message)s', 
                    level=logging.INFO)     # Change level to DEBUG if needed

def run_batch_simulation(simulator_input, simulator_config, custom_config, batch_steps=BATCHS):
    logging.info("Running simulation with batch mode!")
    
    logging.info("Starting simulation...")
    start = time.time()
    for batch in range(0, batch_steps):
        logging.info(f"({batch+1}/{batch_steps}) Executing simulation...")
        execute_simulator(input_parameters=simulator_input, config_parameters=simulator_config, custom_parameters=custom_config)
    logging.info("Ended batch simulation")
    logging.info(f"Elapsed time: {np.round(time.time() - start, decimals=4)} seconds.")
    
    compute_mean_result(simulator_config)

def compute_mean_result(simulator_config):
    global_df = pd.DataFrame()
    output_folder = os.path.join('output', simulator_config['output_folder'])
    for subfolder in os.listdir(output_folder):
        if os.path.isdir(os.path.join(output_folder, subfolder)):
            # Get the paths to the csv files
            data_path = os.path.join(output_folder, subfolder, 'data')
            csv_path = os.path.join(data_path, os.listdir(data_path)[0])

            # Read the csv file into a dataframe
            df = pd.read_csv(csv_path)
            df = df.drop(['time[s]'], axis=1)
            means = df.mean()
            global_df = pd.concat([global_df, means.to_frame().T], axis=0)

    # Save the global dataframe to a csv file
    mean_simulation_csv = os.path.join(output_folder, 'output-simulations.csv')
    global_df.to_csv(mean_simulation_csv, index=False)
    
    # Compute global result output
    means_output_df = global_df.mean()
    result_df = pd.DataFrame({'Column Name': means_output_df.index, 'Mean Value': means_output_df.values})
    result_df = result_df.set_index('Column Name').transpose()
    
    output_result_csv = os.path.join(output_folder, 'output-result.csv')
    result_df.to_csv(output_result_csv, index=False)
    logging.info("Batch simulation complete!")

if __name__ == "__main__":
    
    # Setup config!
    simulator_input = INPUT_PARAMETERS.copy()
    simulator_input['Simulation_Time'] = 0.5*60
    simulator_input['Users'] = 50
    
    simulator_config = CONFIG_PARAMETERS.copy()
    simulator_config['save_output'] = True      # Save the run output on output folder
    simulator_config['show_plots'] = False      # Show output plots
    simulator_config['output_folder'] = BATCH_NAME
    simulator_config['algorithm'] = "elighthouse"
    
    custom_config = {}
    custom_config['user_report_position'] = 4   # For each four timeSteps, the users updates position
    custom_config['startup_max_tokens'] = 2     # TimeSlots to startup a FemtoCell
    custom_config['poweroff_unused_cell'] = 2   # TimeSlots to poweroff an unused Cell
    
    run_batch_simulation(simulator_input=simulator_input,
                         simulator_config=simulator_config,
                         custom_config=custom_config)