__author__ = "Enrique Fernandez Sanchez (efernandez@e-lighthouse.com)"
__credits__ = ["Enrique Fernandez Sanchez"]
__version__ = "1.2"
__maintainer__ = "Enrique Fernandez Sanchez"
__email__ = "efernandez@e-lighthouse.com"
__status__ = "Validated"

import numpy as np
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
import matplotlib.pyplot as plt

from simulator.launch import logger
from simulator.solar_harvesting import SolarPanel, Weather
from simulator.algorithm_tecno_new import PoF_simulation_ELighthouse_TecnoAnalysis

class PoF_simulation_ELighthouse_TecnoAnalysisPlots:

    def __init__(self, sim_alg: PoF_simulation_ELighthouse_TecnoAnalysis):
        self.alg = sim_alg

    def plot_output(self, sim_times, timeStep, is_gui: bool = False, show_plots: bool = True, kwargs: dict = {}):
        """ Override Show Plot Output

        Args:
            sim_times (_type_): _description_
            show_plots (bool, optional): _description_. Defaults to True.
        """
        
        alg = self.alg
        fig_size = kwargs.get('fig_size', (12, 8))
        
        
        # Battery dead?
        if alg.timeIndex_first_battery_dead != 0:
            alg.first_batt_dead_s = (alg.timeIndex_first_battery_dead*timeStep)
            alg.last_batt_dead_s = (alg.timeIndex_last_battery_dead*timeStep)
            alg.remaining_batt = np.count_nonzero(np.round(alg.battery_vector[0])) - alg.NMacroCells
            logger.info(f"Last Battery dead at timeIndex: {alg.timeIndex_last_battery_dead} ({alg.last_batt_dead_s/60} min)")
            logger.info(f"First Battery dead at timeIndex: {alg.timeIndex_first_battery_dead} ({alg.first_batt_dead_s/60} min)")
            logger.info(f"Remaining batteries {alg.remaining_batt} of {len(alg.battery_vector[0]) - alg.NMacroCells}.")
        
        # Compute %'s
        # self.is_in_femto -> 1 == associated with femto, -> 2 == associated with macro, -> 0 == no on femto area
        sum_served_femto = 0    # % that a user is in femto area, and its associated
        sum_in_area = 0         # % that a user is in femto area (associated or not)          
        sum_time_served = 0     # % that user is in femto and its has been associated with a femto                
        for user in range(0, len(alg.NUsers)):
            t_served_femto = np.count_nonzero(alg.is_in_femto[user] == 1)
            sum_served_femto += (t_served_femto) / (len(sim_times))
            t_in_area = np.count_nonzero(alg.is_in_femto[user] == 1) + np.count_nonzero(alg.is_in_femto[user] == 2)
            sum_in_area += (t_in_area) / (len(sim_times))
            try:
                sum_time_served += (t_served_femto) / (t_in_area)
            except:
                sum_time_served += 0

        alg.per_served_femto = np.round(((1/len(alg.NUsers)) * sum_served_femto) * 100, 3)
        alg.per_in_area = np.round(((1/len(alg.NUsers)) * sum_in_area) * 100, 3)
        alg.per_time_served = np.round(((1/len(alg.NUsers)) * sum_time_served) * 100, 3)
        
        logger.info(f"% in area & served by femto: {alg.per_served_femto} %")
        logger.info(f"% in area of femto: {alg.per_in_area} %")
        logger.info(f"% of inside time, when user is in area and associated with femto : {alg.per_time_served} %")
        
        # User Traffic
        fig_user_traffic, ax = plt.subplots()
        self.list_figures.append((fig_user_traffic, "user-traffic"))    # In Order to save the figure on output folder
        
        metric = 0  # Default traffic
        for user in range(0, len(self.NUsers)):
            user_traffic = np.asarray([self.X_user[t][user][metric] for t in range(len(sim_times))])
            ax.plot(sim_times, user_traffic/10e6, label=f'User {user}')
        ax.legend(fontsize='x-small', ncols=3)
        ax.set_title(f'Traffic for each user')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Throughput [Mb/s]')
        
        # Batteries in use for each timeStep
        battery_charging = []
        for timeIndex in alg.battery_state:
            # 0 = nothing; 1 = charging; 2 = discharging; 3 = discharging & charging.
            #count_3 = np.count_nonzero(timeIndex == 3.0)
            #count_1 = np.count_nonzero(timeIndex == 1.0)
            count_2 = np.count_nonzero(timeIndex == 2.0)
            #battery_charging.append(count_3 + count_1)
            battery_charging.append(count_2)
            
        fig_battery_charging, ax = plt.subplots()
        alg.list_figures.append((fig_battery_charging, "discharging-cells"))    # In Order to save the figure on output folder
        ax.plot(sim_times, battery_charging, label="Discharging Cells")
        ax.legend()
        ax.set_title("Discharging Battery Cells")
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Number of cells')
        
        # Battery capacity
        fig_batt_capacity, ax = plt.subplots()
        alg.list_figures.append((fig_batt_capacity, "batt-capacity")) 
        ax.axhline(y=3300, color='r',label="Max. capacity")
        for bar in range(0, len(alg.battery_vector[0])):
            if bar >= alg.NMacroCells:
                ax.bar(int(bar), alg.battery_vector[0][bar]*1000)
        ax.legend()
        ax.set_title("Battery Capacity")
        ax.set_xlabel("Femto cell number")
        ax.set_ylabel("Capacity [mAh]")
        
        # New Figures
        if alg.use_harvesting:
            fig_battery_mean_harvesting, ax = plt.subplots()
            alg.list_figures.append((fig_battery_mean_harvesting, "battery_mean_harvesting"))
            ax.plot(sim_times, alg.battery_mean_values, '-', label='Hybrid PoF & Solar', color="tab:red")
            ax.plot(sim_times, alg.battery_mean_values - alg.battery_mean_harvesting, '--', label='Only PoF', color="tab:blue")
            ax.axhline(y=3.3, color='tab:green',label="Max. battery capacity")
            ax.set_xlabel('Time [s]')
            ax.set_ylabel('Battery capacity [Ah]')
            #ax.set_title('Mean Battery Capacity of the System')
            ax.legend()
            
            fig_battery_acc_harvesting, ax = plt.subplots()
            alg.list_figures.append((fig_battery_acc_harvesting, "battery_accumulative_harvesting"))
            ax.plot(sim_times, alg.battery_mean_harvesting, label='Accumulative battery harvesting')
            ax.set_xlabel('Time [s]')
            ax.set_ylabel('Battery capacity [Ah]')
            ax.set_title('Accumulative battery harvesting')
            ax.legend()
           
        ## Throughput
        fig_throughput, ax = plt.subplots()
        alg.list_figures.append((fig_throughput, 'output-throughput'))
        ax.plot(sim_times, alg.output_throughput[0]/10e6, label="Macro Cells")
        ax.plot(sim_times, alg.output_throughput[1]/10e6, label="Femto Cells")
        ax.plot(sim_times, alg.live_throughput/10e6, label="Total")
        ax.legend()
        ax.set_title("Throughput Downlink. System with batteries")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel('Throughput [Mb/s]')
        
        ## Throughput no battery
        fig_throughput_no_batt, ax = plt.subplots()
        alg.list_figures.append((fig_throughput_no_batt, 'output-throughput-no-batt'))
        ax.plot(sim_times, alg.output_throughput_no_batt[0]/10e6, label="Macro Cells")
        ax.plot(sim_times, alg.output_throughput_no_batt[1]/10e6, label="Femto Cells")
        ax.plot(sim_times, alg.output_throughput_no_batt[2]/10e6, label="Femto Cells overflow")
        ax.plot(sim_times, alg.live_throughput_NO_BATTERY/10e6, label="Total")
        ax.legend()
        ax.set_title("Throughput Downlink. System without batteries")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel('Throughput [Mb/s]')
        
        ## Only Macro
        fig_throughput_only_macro, ax = plt.subplots()
        alg.list_figures.append((fig_throughput_only_macro, 'output-throughput-only-macro'))
        ax.plot(sim_times, alg.output_throughput_only_macro/10e6, label="Macro Cells")
        ax.legend()
        ax.set_title("Throughput Downlink. System with only Macro Cells")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel('Throughput [Mb/s]')
        




        # Plot associations of users to cells. Two plots:
        # 1. Femto cells regions
        # 2. Macro cells regions

        last_user_to_bs_assoc = alg.association_vector[0, :]

        # regions_fem = self.Regions[self.NMacroCells:]
        # regions_mac = self.Regions[:self.NMacroCells]

        users_pos = np.array([
            [alg.user_list[user]["v_x"][-1], alg.user_list[user]["v_y"][-1]]
            for user in range(len(alg.NUsers))
        ])

        users_in_fem = np.zeros(len(last_user_to_bs_assoc), dtype=int)
        users_in_mac = np.zeros(len(last_user_to_bs_assoc), dtype=int)

        for user, bs in enumerate(last_user_to_bs_assoc):
            if bs >= alg.NMacroCells:
                users_in_fem[user] = 1
            else:
                users_in_mac[user] = 1


        p2p_lines_fem = []
        p2p_lines_mac = []
        users_fem = []
        users_mac = []



        for user, bs in enumerate(last_user_to_bs_assoc):
            user_x = alg.user_list[user]["v_x"][-1]
            user_y = alg.user_list[user]["v_y"][-1]
            bs_x = alg.BaseStations[int(bs)][0]
            bs_y = alg.BaseStations[int(bs)][1]

            x_coords = [user_x, bs_x]
            y_coords = [user_y, bs_y]

            if bs >= alg.NMacroCells:
                p2p_lines_fem.append((x_coords, y_coords))
                users_fem.append((user_x, user_y))
            else:
                p2p_lines_mac.append((x_coords, y_coords))
                users_mac.append((user_x, user_y))





        # Paint the regions (bs areas)
        region_config = {
            "alpha": 0.3,
            "edgecolor": 'black',
            "linewidth": 0.5,
            "linestyle": '-',
        }
        def paint_regions(_regions, _ax):
            for i in range(len(_regions) - 1, -1, -1):
                _region = _regions[i]
                if isinstance(_region, Polygon):
                    x, y = _region.exterior.coords.xy
                    _ax.fill(x, y, **region_config)
                elif isinstance(_region, MultiPolygon):
                    for reg in _region.geoms:
                        x, y = reg.exterior.coords.xy
                        _ax.fill(x, y, **region_config)
                elif isinstance(_region, GeometryCollection):
                    for geom in _region.geoms:
                        if isinstance(geom, Polygon):
                            x, y = geom.exterior.coords.xy
                            _ax.fill(x, y, **region_config)
                else:
                    x, y = _region.exterior.coords.xy
                    _ax.fill(x, y, **region_config)

        node_config = {
            "marker": "o",
            "s": 30,
            "color": 'blue',
            "alpha": 1,
        }
        def paint_node(x, y, _ax):
            _ax.scatter(x, y, color='black', s=10, marker='o')

        def paint_association(line, _ax):
            _ax.plot(line[0], line[1], color='green', linewidth=0.8)

        def paint_user(x, y, _ax):
            _ax.scatter(x, y, color='red', s=20, marker='+', linewidths=2)

        def paint_user_no_active(x, y, _ax):
            _ax.scatter(x, y, color='blue', s=5, marker='+', linewidths=0.5)




        fig_user_assoc_only_femto, ax_fem = plt.subplots()
        alg.list_figures.append((fig_user_assoc_only_femto, 'output-last-user-association-only-femto'))
        ax_fem.set_title('Last User Association - Femto Cells')
        ax_fem.scatter([], [], label="User in FemtoCell", color='red', s=20, marker='+', linewidths=2)
        ax_fem.scatter([], [], label="User in MacroCell", color='blue', s=20, marker='+', linewidths=2)
        ax_fem.scatter([], [], label="FemtoCell", color='black', s=10, marker='o')
        ax_fem.legend()
        ax_fem.axis('off')
        fig_user_assoc_only_femto.tight_layout()
        # ax_fem.set_xlabel('X [km]')
        # ax_fem.set_ylabel('Y [km]')


        fig_user_assoc_only_macro, ax_mac = plt.subplots()
        alg.list_figures.append((fig_user_assoc_only_macro, 'output-last-user-association-only-macro'))
        ax_mac.set_title('Last User Association - Macro Cells')
        ax_mac.scatter([], [], label="User in MacroCell", color='red', s=20, marker='+', linewidths=2)
        ax_mac.scatter([], [], label="User in FemtoCell", color='blue', s=20, marker='+', linewidths=2)
        ax_mac.scatter([], [], label="MacroCell", color='black', s=10, marker='o')
        ax_mac.legend()
        ax_mac.axis('off')
        fig_user_assoc_only_macro.tight_layout()
        # ax_mac.set_xlabel('X [km]')
        # ax_mac.set_ylabel('Y [km]')

        # Loop over Femto BSs
        for i in range(alg.NMacroCells, len(alg.BaseStations)):
            region = alg.Regions[i]
            paint_regions([region], ax_fem)  # wrap single _region in list
            bs_x, bs_y = alg.BaseStations[i][:2]
            paint_node(bs_x, bs_y, ax_fem)

            for user, bs in enumerate(last_user_to_bs_assoc):
                if bs == i:
                    paint_user(*users_pos[user], ax_fem)

        for line in p2p_lines_fem:
            paint_association(line, ax_fem)

        for user in users_fem:
            paint_user(*user, ax_fem)

        for user in users_mac:
            paint_user_no_active(*user, ax_fem)

        # Loop over Macro BSs
        for i in range(self.NMacroCells):
            region = alg.Regions[i]
            paint_regions([region], ax_mac)
            bs_x, bs_y = alg.BaseStations[i][:2]
            paint_node(bs_x, bs_y, ax_mac)

            for user, bs in enumerate(last_user_to_bs_assoc):
                if bs == i:
                    paint_user(*users_pos[user], ax_mac)

        for line in p2p_lines_mac:
            paint_association(line, ax_mac)

        for user in users_mac:
            paint_user(*user, ax_mac)

        for user in users_fem:
            paint_user_no_active(*user, ax_mac)






        # Get the context_class method
        alg.plot_output(sim_times=sim_times, show_plots=show_plots, is_gui=is_gui, fig_size=fig_size)

