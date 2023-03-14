import logging, time, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from simulator.launch import INPUT_PARAMETERS, CONFIG_PARAMETERS
from simulator.launch import execute_simulator

logging.basicConfig(format='%(asctime)s %(levelname)s:%(module)s:%(funcName)s: %(message)s', 
                    level=logging.INFO)     # Change level to DEBUG if needed

######
# Utils
######
def generate_simulation_points(start, end, size):
    # Generate a sorted array of unique random integers of the given size
    # between the start and end numbers (inclusive) using NumPy.
    array = np.random.choice(np.arange(start, end+1), size=size, replace=False)
    array.sort()
    return array

def plot_parametric(output_field, title, parametric_run, show_plots):
    parametric_folder = os.path.join('output', parametric_run)
    fig, ax = plt.subplots()
    for x in os.listdir(parametric_folder):
        run_folder = os.path.join(parametric_folder, x)
        csv = os.path.join(run_folder, f'data/{x}-output.csv')        
        df = pd.read_csv(csv)
        ax.plot(df['time[s]'], df[output_field], label=x)
     
    plt.title(title)
    plt.ylabel("Throughputs (Mb/s)")
    plt.xlabel("Time [s]")  
    plt.grid()
    plt.legend(loc="upper left")
    if show_plots:
        plt.show(block=False)
        input("hit [enter] to close figures")
        plt.close()

######
# Parametric Sweep Main function!
######
def paremetric_sweep(show_plots: bool = True):
    logging.info("Launching Parametric Sweep!")
    
    #######
    # Parametric sweep number of lasers
    parametric_name = 'parametric_lasers-1'
    field = 'numberOfLasers'
    low_value = 2
    high_value = 10
    #######
    
    logging.info(f"Field to simulate: {field}")
    field_list = generate_simulation_points(start=low_value,
                                            end=high_value,
                                            size=high_value-low_value)
    logging.info(f"Selected values to simulate: {field_list}")
    
    logging.info("Configuring config_simulator...")
    sim_config = CONFIG_PARAMETERS.copy()
    sim_config['use_user_list'] = True
    sim_config['save_output'] = True
    sim_config['show_plots'] = False
    sim_config['use_nice_setup'] = True
    sim_config['output_folder'] = parametric_name
    
    logging.info(f"Starting parametric sweep for field: {field}")
    start = time.time()
    
    pos = 0
    for value in field_list:
        pos += 1
        # For each point of the parametric simulation...
        sim_input = INPUT_PARAMETERS.copy()
        sim_input[field] = value
        
        logging.info(f"({pos}/{len(field_list)}) Execute simulation: {field}-{value}")
        execute_simulator(input_parameters=sim_input,
                          config_parameters=sim_config,
                          run_name=f"{field}-{value}")
    logging.info("Ended parametric sweep")
    logging.info(f"Elapsed time: {np.round(time.time() - start, decimals=4)} seconds.")
    
    plot_parametric(output_field="throughput_no_battery[mbps]",
                    title=f"Parametric swipe of {field}. Plot of Throughput without battery",
                    parametric_run=parametric_name,
                    show_plots=show_plots)
    
if __name__ == "__main__":
    paremetric_sweep()