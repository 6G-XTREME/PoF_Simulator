__author__ = "Gabriel Otero Perez (gaoterop@it.uc3m.es), Jose-Manuel Martinez-Caro (jmmartinez@e-lighthouse.com), Enrique Fernandez Sanchez (efernandez@e-lighthouse.com)"
__credits__ = ["Gabriel Otero Perez", "Jose-Manuel Martinez-Caro", "Enrique Fernandez Sanchez"]
__version__ = "1.1"
__maintainer__ = "Jose-Manuel Martinez-Caro"
__email__ = "jmmartinez@e-lighthouse.com"
__status__ = "Validated"

import numpy as np
import pandas as pd
import scipy.io, os
import matplotlib.pyplot as plt
from simulator.launch import logger
import json
from run_simulator_technoeconomics import CONFIG_PARAMETERS

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
        self.NUsers = user_data["users"]
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
        self.colorsBS = basestation_data.get('colorsBS', None)
        # self.RegionsMacrocells = basestation_data['RegionsMacrocells']

        # Power & Battery Parameters
        self.small_cell_consumption_ON = battery_data['small_cell_consumption_ON']
        self.small_cell_consumption_SLEEP = battery_data['small_cell_consumption_SLEEP']
        self.small_cell_current_draw = battery_data['small_cell_current_draw']
        self.small_cell_voltage_range = battery_data['small_cell_voltage_range']
        self.max_energy_consumption_total = battery_data['max_energy_consumption_total']
        self.max_energy_consumption_active = battery_data['max_energy_consumption_active']
        
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


    def start_simulation(self, sim_times, timeStep, text_plot, show_plots: bool = True, speed_plot: float = 0.05):
        pass
    
    def plot_output(self, sim_times, is_gui: bool = False, show_plots: bool = True, fig_size: tuple = None, dpi: int = None):
        """Plot simulation outputs with configurable figure parameters.

        Args:
            sim_times: Array of simulation times
            is_gui: Whether running in GUI mode
            show_plots: Whether to display plots
            fig_size: Optional override for figure size (width, height) in inches
            dpi: Optional override for figure DPI
        """
        # Get figure configuration from CONFIG_PARAMETERS
        fig_config = CONFIG_PARAMETERS.get('figure_config', {})
        fig_size = fig_size or fig_config.get('fig_size', (12, 8))
        dpi = dpi or fig_config.get('dpi', 100)
        line_width = fig_config.get('line_width', 1.5)
        font_size = fig_config.get('font_size', 12)
        tick_size = fig_config.get('tick_size', 10)

        # Set global matplotlib parameters
        plt.rcParams.update({
            'font.size': font_size,
            'axes.labelsize': font_size,
            'axes.titlesize': font_size,
            'xtick.labelsize': tick_size,
            'ytick.labelsize': tick_size,
            'lines.linewidth': line_width
        })

        def format_time_axis(ax, times):
            """Helper function to format time axis based on total duration"""
            total_seconds = times[-1]
            if total_seconds > 2 * 86400:  # More than a day
                ax.set_xlabel('Time [days]')
                # Convert to days and set ticks every 6 hours
                times_days = times / 86400
                ax.set_xticks(np.arange(0, times_days[-1] + 0.25, 0.25))  # 0.25 days = 6 hours
                ax.set_xticklabels([f'{x:.1f}' for x in np.arange(0, times_days[-1] + 0.25, 0.25)])
                return times_days
            elif total_seconds > 2 * 3600:  # More than an hour
                ax.set_xlabel('Time [hours]')
                # Convert to hours and set ticks every hour
                times_hours = times / 3600
                ax.set_xticks(np.arange(0, times_hours[-1] + 1, 1))
                ax.set_xticklabels([f'{int(x)}' for x in np.arange(0, times_hours[-1] + 1, 1)])
                return times_hours
            else:  # Less than an hour
                ax.set_xlabel('Time [seconds]')
                # Set ticks every 10 minutes (600 seconds)
                tick_interval = 600
                ax.set_xticks(np.arange(0, total_seconds + tick_interval, tick_interval))
                ax.set_xticklabels([f'{int(x/60)}' for x in np.arange(0, total_seconds + tick_interval, tick_interval)])
                return times

        # 1
        fig_cell_occupancy, ax = plt.subplots(figsize=fig_size, dpi=dpi)
        self.list_figures.append((fig_cell_occupancy, "cell_occupancy"))
        ax.axhline(y=self.NFemtoCells, color='r', label='Total Small cells')
        ax.step(format_time_axis(ax, sim_times), self.live_smallcell_occupancy, 'g', label='Small cells being used')
        ax.step(format_time_axis(ax, sim_times), self.live_smallcell_overflow, 'b', label='Small cells overflowed')
        ax.legend()
        ax.set_title('Number of small cells under use')
        ax.set_ylabel('Number of cells')

        # 2
        fig_cell_consumption, ax = plt.subplots(figsize=fig_size, dpi=dpi)
        self.list_figures.append((fig_cell_consumption, "cell_consumption"))
        ax.axhline(y=self.max_energy_consumption_total, color='b', label='Max power of lasers [W]')
        ax.axhline(y=self.max_energy_consumption_active, color='r', label='Max consumption of PoF budget [W]')
        ax.step(format_time_axis(ax, sim_times), self.live_smallcell_consumption, 'g', label='Live energy consumption [W], laser + battery', linewidth=line_width)
        ax.legend()
        ax.set_ylim(0, max(max(self.live_smallcell_consumption), self.max_energy_consumption_total, self.max_energy_consumption_active) * 1.1)
        ax.set_title('Live energy consumption')
        ax.set_ylabel('Power consumption (Watts)')

        # 3
        fig_throughput, ax = plt.subplots(figsize=fig_size, dpi=dpi)
        self.list_figures.append((fig_throughput, "throughput"))
        ax.plot(format_time_axis(ax, sim_times), self.live_throughput/10e6, label='With battery system')
        ax.plot(format_time_axis(ax, sim_times), self.live_throughput_NO_BATTERY/10e6, 'r--', label='Without battery system')
        ax.plot(format_time_axis(ax, sim_times), self.live_throughput_only_Macros/10e6, 'g:.', label='Only Macrocells')
        ax.legend()
        ax.set_title('Live system throughput [un-smooth]')
        ax.set_ylabel('Throughput [Mb/s]')

        # 3, filtered one
        SMA_WINDOW = 1
        timeIndex = len(sim_times)
        fig_throughput_smooth, ax = plt.subplots(figsize=fig_size, dpi=dpi)
        self.list_figures.append((fig_throughput_smooth, "throughput_smooth"))
        X = format_time_axis(ax, sim_times[timeIndex-(len(self.live_throughput)-(SMA_WINDOW-1))+1:timeIndex])
        Y = np.convolve(self.live_throughput/10e6, np.ones((SMA_WINDOW,))/SMA_WINDOW, mode='valid')
        ax.plot(X, Y[:-1], label='Using PoF & batteries')

        X = format_time_axis(ax, sim_times[timeIndex-(len(self.live_throughput_NO_BATTERY)-(SMA_WINDOW-1))+1:timeIndex])
        Y = np.convolve(self.live_throughput_NO_BATTERY/10e6, np.ones((SMA_WINDOW,))/SMA_WINDOW, mode='valid')
        ax.plot(X, Y[:-1], 'r--', label='Using PoF')

        X = format_time_axis(ax, sim_times[timeIndex-(len(self.live_throughput_only_Macros)-(SMA_WINDOW-1))+1:timeIndex])
        Y = np.convolve(self.live_throughput_only_Macros/10e6, np.ones((SMA_WINDOW,))/SMA_WINDOW, mode='valid')
        ax.plot(X, Y[:-1], 'g--o', label='Only macrocells')

        ax.legend()
        ax.set_title('Live system throughput')
        ax.set_ylabel('Throughput [mbps]')

        # 5
        fig_battery_mean, ax = plt.subplots(figsize=fig_size, dpi=dpi)
        self.list_figures.append((fig_battery_mean, "battery_mean"))
        ax.plot(format_time_axis(ax, sim_times), self.battery_mean_values, label='Battery mean capacity')
        ax.axhline(y=3.3, color='r',label="Max. voltage battery")
        ax.set_ylabel('Battery capacity [Ah]')
        ax.set_title('Mean Battery Capacity of the System')
        ax.legend()
        
        if show_plots:
            plt.show(block=False)
            if not is_gui:
                plt.pause(0.001)
                input("hit [enter] to close plots and continue")
                plt.close('all')
        else:
            plt.close('all')
        

    def save_run(self, fig_map, sim_times, run_name, output_folder, dpi: int = 200):
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
        if fig_map is not None:
            fig_map.savefig(os.path.join(plot_folder, f'{run_name}-map.png'))
            plt.close(fig_map)
        
        for fig in self.list_figures:
            fig[0].savefig(os.path.join(plot_folder, f'{run_name}-{fig[1]}.png'), dpi=dpi, bbox_inches='tight')
            #plt.close(fig[0])

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
