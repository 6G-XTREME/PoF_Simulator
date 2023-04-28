import os
import logging
import pandas as pd

from simulator.launch import INPUT_PARAMETERS, CONFIG_PARAMETERS
from run_batch import run_batch_simulation

BATCH_STEPS = 20
PARAMETER = "Users"
NAME = PARAMETER.lower() + "-parametric"
VALUES_TO_VISIT = [30, 60, 90, 120, 160, 180]   # p.e Users
SIMULATION_TIME = 30                            # in minutes

logging.basicConfig(format='%(asctime)s %(levelname)s:%(module)s:%(funcName)s: %(message)s', 
                    level=logging.INFO)     # Change level to DEBUG if needed

def run_parametric():
    logging.info("Starting Parametric Simulation!")
    
    if os.path.exists(os.path.join("output", NAME)):
        logging.error("Unable to start simulation, a parametric simulation exists with that name.")
        raise Exception
    
    # Default config!
    simulator_input = INPUT_PARAMETERS.copy()
    simulator_input['Simulation_Time'] = SIMULATION_TIME*60
    simulator_input['Users'] = 90
    
    simulator_config = CONFIG_PARAMETERS.copy()
    simulator_config['save_output'] = True      # Save the run output on output folder
    simulator_config['show_plots'] = False      # Show output plots
    
    simulator_config['algorithm'] = "elighthouse"
    
    custom_config = {}
    custom_config['user_report_position'] = 8   # For each four timeSteps, the users updates position
    custom_config['startup_max_tokens'] = 8     # TimeSlots to startup a FemtoCell
    custom_config['poweroff_unused_cell'] = 8   # TimeSlots to poweroff an unused Cell
    
    for pos in range(0, len(VALUES_TO_VISIT)):
        simulator_input[PARAMETER] = VALUES_TO_VISIT[pos]
        custom_config[PARAMETER] = VALUES_TO_VISIT[pos]
        simulator_config['output_folder'] = NAME + "-" + str(pos)
        
        run_batch_simulation(simulator_input=simulator_input,
                             simulator_config=simulator_config,
                             custom_config=custom_config,
                             batch_steps=BATCH_STEPS)
        logging.info(f"Parametric step complete ({pos+1}/{len(VALUES_TO_VISIT)})")
    
    # Ended simulation, compute parametric output!
    base_folder = "./output"

    # Find the subfolder that starts with the name variable
    for subfolder in sorted(os.listdir(base_folder)):
        if subfolder.startswith(NAME) and subfolder != NAME:
            # Create a new folder inside the output folder with the 
            # name variable as the name
            new_folder = os.path.join(base_folder, NAME)
            if not os.path.exists(new_folder):
                os.mkdir(new_folder)
            # Move the subfolder to the new folder
            os.rename(os.path.join(base_folder, subfolder), 
                      os.path.join(new_folder, os.path.basename(subfolder)))

    # Process the parametric results and saves in a DataFrame
    dfs = []
    for user_folder in sorted(os.listdir(os.path.join(base_folder, NAME))):
        result_file = os.path.join(base_folder, NAME, user_folder, 'output-result.csv')
        df = pd.read_csv(result_file)
        dfs.append(df)

    result_df = pd.concat(dfs)
    result_df.insert(0, PARAMETER, VALUES_TO_VISIT)

    # Save the concatenated dataframe to a new file
    output_path = os.path.join(base_folder, NAME, "output-parametric.csv")
    result_df.to_csv(output_path, index=False)
    logging.info("Ended parametric simulation.")
    
if __name__ == "__main__":
    run_parametric()