import numpy as np
import pandas as pd
import scipy.io, os
import matplotlib.figure as Figure
import matplotlib.pyplot as plt

from simulator.launch import logger

class Contex_Config():
    Simulation_Time: int
    
    # BaseStation Parameters
    BaseStations: dict
    Regions: list
    NMacroCells: int
    NFemtoCells: int
    association_vector: np.array
    association_vector_overflow_alternative: np.array
    colorsBS: list
    baseStation_users: np.array
    overflown_from: np.array
    temporal_association_vector: np.array
    
    # User Parameters
    NUsers: int
    user_list: np.array
    user_pos_plot: np.array
    user_association_line: np.array
    
    # Power & Battery Parameters
    battery_vector: np.array
    battery_state: np.array
    battery_capacity: float
    battery_mean_values: np.array
    small_cell_consumption_ON: float
    small_cell_consumption_SLEEP: float
    small_cell_current_draw: float
    small_cell_voltage_range: np.array
    max_energy_consumption: float
    
    # Transmitting Power Data
    noise: float
    alpha_loss: float
    PMacroCells: float
    PFemtoCells: float
    MacroCellDownlinkBW: float
    FemtoCellDownlinkBW: float
    
    # Output data
    live_smallcell_occupancy: np.array
    live_smallcell_overflow: np.array
    live_smallcell_consumption: np.array
    live_throughput: np.array
    live_throughput_NO_BATTERY: np.array
    live_throughput_only_Macros: np.array
    battery_mean_values: np.array
    
    # Figures
    list_figures: list
    
    def __init__(self, sim_times, basestation_data: dict, user_data: dict, battery_data: dict, transmit_power_data: dict) -> None:
        # User Parameters
        self.NUsers = user_data["number_users"]
        self.user_list = user_data['user_list']
        self.user_pos_plot = user_data['user_pos_plot']
        self.user_association_line = user_data['user_association_line']
        
        # BaseStation Parameters
        self.BaseStations = basestation_data['BaseStations']
        self.Regions = basestation_data['Regions']
        self.NMacroCells = basestation_data['NMacroCells']
        self.NFemtoCells = basestation_data['NFemtoCells']
        self.association_vector = np.zeros((1, len(self.NUsers)))
        self.association_vector_overflow_alternative = np.zeros((1, len(self.NUsers)))
        self.colorsBS = basestation_data['colorsBS']

        # Power & Battery Parameters
        self.small_cell_consumption_ON = battery_data['small_cell_consumption_ON']
        self.small_cell_consumption_SLEEP = battery_data['small_cell_consumption_SLEEP']
        self.small_cell_current_draw = battery_data['small_cell_current_draw']
        self.small_cell_voltage_range = battery_data['small_cell_voltage_range']
        self.max_energy_consumption = battery_data['max_energy_consumption']
        
        self.battery_capacity = battery_data['battery_capacity']
        self.battery_vector = self.battery_capacity * np.ones((1, self.NMacroCells + self.NFemtoCells))
        self.battery_mean_values = np.zeros(len(sim_times)) + self.battery_capacity
        
        # Transmitting Power Data
        self.noise = transmit_power_data['noise']
        self.alpha_loss = transmit_power_data['alpha_loss']
        self.PMacroCells = transmit_power_data['PMacroCells']
        self.PFemtoCells = transmit_power_data['PFemtoCells']
        self.MacroCellDownlinkBW = transmit_power_data['MacroCellDownlinkBW']
        self.FemtoCellDownlinkBW = transmit_power_data['FemtoCellDownlinkBW']
        
        # Start output data
        self.live_throughput = np.zeros(len(sim_times))
        self.live_throughput_NO_BATTERY = np.zeros(len(sim_times))
        self.live_throughput_only_Macros = np.zeros(len(sim_times))
        self.live_smallcell_consumption = np.zeros(len(sim_times))
        self.live_smallcell_occupancy = np.zeros(len(sim_times))
        self.live_smallcell_overflow = np.zeros(len(sim_times))
        
        self.Simulation_Time = sim_times[-1]
        self.list_figures = []        
        pass
    
    def start_simulation(self, sim_times, timeStep, text_plot, show_plots: bool = True, speed_plot: float = 0.05):
        pass
    
    def plot_output(self, sim_times, show_plots: bool = True):
        # 1
        fig_cell_occupancy, ax = plt.subplots()
        self.list_figures.append((fig_cell_occupancy, "cell_occupancy"))
        ax.axhline(y=self.NFemtoCells, color='r', label='Total Small cells')
        ax.plot(sim_times, self.live_smallcell_occupancy, 'g', label='Small cells being used')
        ax.plot(sim_times, self.live_smallcell_overflow, 'b', label='Small cells overflowed')
        #ax.text(0, NFemtoCells - 1, f"Phantom Cells ON: {NFemtoCells - 1}")
        #ax.axhline(y=numberOfLasers-1, label=f"Max. Lasers:")
        ax.legend()
        ax.set_title('Number of small cells under use')

        # 2
        fig_cell_consumption, ax = plt.subplots()
        self.list_figures.append((fig_cell_consumption, "cell_consumption"))
        ax.axhline(y=self.small_cell_consumption_ON * self.NFemtoCells, color='r', label='Total always ON consumption [W]')
        ax.plot(sim_times, self.live_smallcell_consumption, 'g', label='Live energy consumption [W]')
        #ax.text(1, small_cell_consumption_ON * NFemtoCells - 1, f"Energy consumption (Active Femtocells): {small_cell_consumption_ON * NFemtoCells - 1} W")
        #ax.text(1, small_cell_consumption_ON * NFemtoCells - 3, f"Energy consumption (Idle Femtocells): {small_cell_consumption_ON * NFemtoCells - 3} W")
        #ax.text(1, small_cell_consumption_ON * NFemtoCells - 5, f"Energy consumption (Total Femtocells): {small_cell_consumption_ON * NFemtoCells - 5} W")
        ax.legend()
        ax.set_title('Live energy consumption')

        # 3
        fig_throughput, ax = plt.subplots()
        self.list_figures.append((fig_throughput, "throughput"))
        ax.plot(sim_times, self.live_throughput/10e6, label='With battery system')
        ax.plot(sim_times, self.live_throughput_NO_BATTERY/10e6, 'r--', label='Without battery system')
        ax.plot(sim_times, self.live_throughput_only_Macros/10e6, 'g:.', label='Only Macrocells')
        ax.legend()
        ax.set_title('Live system throughput')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Throughput [Mb/s]')

        # 3, filtered one
        #SMA_WINDOW = input_parameters['SMA_WINDOW']
        SMA_WINDOW = 5
        timeIndex = len(sim_times)
        fig_throughput_smooth, ax = plt.subplots()
        self.list_figures.append((fig_throughput_smooth, "throughput_smooth"))
        X = sim_times[timeIndex-(len(self.live_throughput)-(SMA_WINDOW-1))+1:timeIndex]/3600
        Y = np.convolve(self.live_throughput/10e6, np.ones((SMA_WINDOW,))/SMA_WINDOW, mode='valid')
        ax.plot(X, Y[:-1], label='With battery system')

        X = sim_times[timeIndex-(len(self.live_throughput_NO_BATTERY)-(SMA_WINDOW-1))+1:timeIndex]/3600
        Y = np.convolve(self.live_throughput_NO_BATTERY/10e6, np.ones((SMA_WINDOW,))/SMA_WINDOW, mode='valid')
        ax.plot(X, Y[:-1], 'r--', label='Without battery system')

        X = sim_times[timeIndex-(len(self.live_throughput_only_Macros)-(SMA_WINDOW-1))+1:timeIndex]/3600 
        Y = np.convolve(self.live_throughput_only_Macros/10e6, np.ones((SMA_WINDOW,))/SMA_WINDOW, mode='valid')
        ax.plot(X, Y[:-1], 'g--o', label='Only Macrocells')

        ax.legend()
        ax.set_title('Live system throughput [smooth]')
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Throughput (Mb/s)')

        # 4
        #fig, ax = plt.subplots()
        #for b in range(NFemtoCells):
        #    ax.bar(NMacroCells + b + 1, battery_vector[0, NMacroCells + b], color='b')
        #ax.set_title('Live battery state')

        # 5
        fig_battery_mean, ax = plt.subplots()
        self.list_figures.append((fig_battery_mean, "battery_mean"))
        ax.plot(sim_times, self.battery_mean_values, label='Battery mean capacity')
        ax.axhline(y=3.3, color='r',label="Max. voltage battery")
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Battery capacity [Ah]')
        ax.set_title('Mean Battery Capacity of the System')
        ax.legend()
        
        if show_plots:
            plt.show(block=False)
            input("hit [enter] to close plots and continue")
            plt.close('all')
        
        pass
    
    def save_run(self, fig_map, sim_times, run_name, output_folder):
        logger.info("Saving output data...")
        
        # Save results to files as csv!
        root_folder = os.path.abspath(__file__ + os.sep + os.pardir + os.sep + os.pardir)
        
        if output_folder != None:
            output_folder = os.path.join(root_folder, 'output', output_folder)
        else:
            output_folder = os.path.join(root_folder, 'output')
        
        os.makedirs(output_folder, exist_ok=True)
        logger.info(f"Output folder: {output_folder}")

        run_folder = os.path.join(output_folder, run_name)
        os.makedirs(run_folder, exist_ok=True)
        logger.info(f"Run folder: {run_folder}")

        # Create CSV and Plot folders
        data_folder = os.path.join(run_folder, 'data')
        plot_folder = os.path.join(run_folder, 'plot')
        os.makedirs(data_folder, exist_ok=True)
        os.makedirs(plot_folder, exist_ok=True)

        # Save CSV foreach plot
        df_output = pd.DataFrame({'time[s]': sim_times,
                                  'small_cell_ocupacy': self.live_smallcell_occupancy,
                                  'small_cell_overflow': self.live_smallcell_overflow,
                                  'small_cell_consumption[W]': self.live_smallcell_consumption,
                                  'throughput[mbps]': self.live_throughput/10e6,
                                  'throughput_no_battery[mbps]': self.live_throughput_NO_BATTERY/10e6,
                                  'throughput_only_macro[mbps]': self.live_throughput_only_Macros/10e6,
                                  'battery_mean[Ah]': self.battery_mean_values})
        df_output = df_output.assign(NMacroCells=self.NMacroCells)
        df_output = df_output.assign(NFemtoCells=self.NFemtoCells)
        df_output.to_csv(os.path.join(data_folder, f'{run_name}-output.csv'), index=False)
        df_output.to_json(os.path.join(data_folder, f'{run_name}-output.json'), orient="index", indent=4)
    
        # Save figures to output folder
        fig_map.savefig(os.path.join(plot_folder, f'{run_name}-map.png'))
        plt.close(fig_map)
        
        for fig in self.list_figures:
            fig[0].savefig(os.path.join(plot_folder, f'{run_name}-{fig[1]}.png'))
            plt.close(fig[0])

        # Copy user_list.mat [replicability of run]
        user_list_output_mat_path = os.path.join(run_folder, f'{run_name}-user_list.mat')
        scipy.io.savemat(user_list_output_mat_path, {'user_list': self.user_list, 'Simulation_Time': self.Simulation_Time})
        
        # Copy nice_setup.mat [replicability of run]
        nice_setup_mat_path = os.path.join(run_folder, f'{run_name}-nice_setup.mat')
        nice_setup_struct = {
            'BaseStations': self.BaseStations,
            'NFemtoCells': self.NFemtoCells,
            'NMacroCells': self.NMacroCells,
        }
        scipy.io.savemat(nice_setup_mat_path, nice_setup_struct)
        logger.info("Succesfully saved output files")
        
        pass
    
    def update_live_plots(self):
        """
            On MATLAB was implemented, but on Python throws an error
        """
        #CHECK
        #live_occupancy_plot.set_data(sim_times[:timeIndex], live_smallcell_occupancy)
        #max_occupancy_plot.set_data([0, sim_times[timeIndex]], [NFemtoCells, NFemtoCells])
        #used.set_text('Phantom Cells in ON state: {}'.format(live_smallcell_occupancy[timeIndex]))
        
        # PLOT THINGS!!
        # Update total consumption plot
        # live_consumption_plot.set_data(sim_times[0:timeIndex], live_smallcell_consumption)
        # max_consumption_plot.set_data([0, sim_times[timeIndex]], [small_cell_consumption_ON * NFemtoCells, small_cell_consumption_ON * NFemtoCells])
        # consuming_ON.set_text(f"Energy consumption (Active Femtocells): {live_smallcell_occupancy[timeIndex] * small_cell_consumption_ON} W")
        # consuming_SLEEP.set_text(f"Energy consumption (Idle Femtocells): {(NFemtoCells - live_smallcell_occupancy[timeIndex]) * small_cell_consumption_SLEEP} W")
        # consuming_TOTAL.set_text(f"Energy consumption (Total Femtocells): {live_smallcell_consumption[timeIndex]} W")
        #
        # # Update system throughput plot
        # live_throughput_plot.set_data(sim_times[timeIndex - (len(live_throughput) - (SMA_WINDOW - 1)) + 1:timeIndex], 
        #                             np.convolve(live_throughput / 10e6, np.ones((SMA_WINDOW,))/SMA_WINDOW, mode='valid'))
        # live_throughput_NO_BATTERY_plot.set_data(sim_times[timeIndex - (len(live_throughput_NO_BATTERY) - (SMA_WINDOW - 1)) + 1:timeIndex], 
        #                                         np.convolve(live_throughput_NO_BATTERY / 10e6, np.ones((SMA_WINDOW,))/SMA_WINDOW, mode='valid'))
        # live_throughput_only_Macros_plot.set_data(sim_times[timeIndex - (len(live_throughput_only_Macros) - (SMA_WINDOW - 1)) + 1:timeIndex], 
        #                                         np.convolve(live_throughput_only_Macros / 10e6, np.ones((SMA_WINDOW,))/SMA_WINDOW, mode='valid'))
        # Update battery states
        # for b in range(NFemtoCells):
        #     handleToThisBar[b].set_height(battery_vector[NMacroCells + b])
        #     handleToThisBar[b].set_facecolor(battery_color_codes[battery_state[0, NMacroCells+b]])
        pass
