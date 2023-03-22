import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

from simulator.launch import logger
from simulator.context_config import Contex_Config
import simulator.map_utils, simulator.mobility_utils, simulator.user_association_utils, simulator.radio_utils

class PoF_simulation_ELi(Contex_Config):
    # Traffic Vars
    X_macro: np.array
    X_macro_only: np.array
    X_macro_no_batt: np.array
    X_macro_overflow: np.array
    X_femto: np.array
    X_femto_no_batt: np.array
    X_user : np.array
    
    def start_simulation(self, sim_times, timeStep, s_mobility, text_plot, show_plots: bool = True, speed_plot: float = 0.05):
        # Settting up some vars
        self.battery_state = [[] for _ in range(len(sim_times))]
        self.baseStation_users = [[] for _ in range(len(sim_times))]
        self.active_Cells = [[] for _ in range(len(sim_times))]
        self.overflown_from = [[] for _ in range(len(sim_times))]
        
        # Traffic global vars
        self.X_macro = np.zeros((len(sim_times), self.NMacroCells))
        self.X_macro_only = np.zeros((len(sim_times), self.NMacroCells))
        self.X_macro_no_batt = np.zeros((len(sim_times), self.NMacroCells))
        self.X_macro_overflow = np.zeros((len(sim_times), self.NMacroCells))
        self.X_femto = np.zeros((len(sim_times), self.NMacroCells+self.NFemtoCells))
        self.X_femto_no_batt = np.zeros((len(sim_times), self.NMacroCells+self.NFemtoCells))
        
        # Three metrics to save: 
        # 0. Global system with batteries
        # 1. Femto without batteries
        # 2. Only Macro
        metrics = 3
        self.X_user = np.zeros((len(sim_times), len(s_mobility['NB_USERS']), metrics), dtype=float)
        
        logger.info("Starting simulation...")
        start = time.time()
        # Progress Bar
        with tqdm(total=100, desc='Simulating...') as f:
            for timeIndex in range(len(sim_times)):
                if f.n > f.total:
                    break

                # Update progress bar and message
                f.update(100 / len(sim_times))
                f.set_description("%.2f %% completed..." % (timeIndex * 100 / len(sim_times)+1))

                t = sim_times[timeIndex]
                text_plot.set_text('Time (sec) = {:.2f}'.format(t))

                self.algorithm_step(timeIndex=timeIndex, timeStep=timeStep, s_mobility=s_mobility)
                self.compute_statistics_for_plots(timeIndex=timeIndex)                          # Prepare derivate data for plots
                self.update_battery_state(timeIndex=timeIndex, timeStep=timeStep)               # Update battery state for next timeStep
                
                if show_plots:
                    plt.draw()
                    plt.pause(speed_plot)
                
        # Finished
        logger.info("Simulation complete!")
        logger.info(f"Elapsed time: {np.round(time.time() - start, decimals=4)} seconds.")
        return

 
    def algorithm_step(self, timeIndex, timeStep, s_mobility):
        if timeIndex == 0: self.battery_state[timeIndex] = np.zeros(self.NMacroCells+self.NFemtoCells)      # 0 = nothing; 1 = charging; 2 = discharging; 3 = discharging & charging.
        try:
            self.battery_state[timeIndex+1] = np.zeros(self.NMacroCells+self.NFemtoCells)                   # 0 = nothing; 1 = charging; 2 = discharging; 3 = discharging & charging.
            self.baseStation_users[timeIndex+1] = np.zeros(self.NMacroCells+self.NFemtoCells)               # Number of users in each base station.
            self.active_Cells[timeIndex+1] = np.zeros(self.NMacroCells+self.NFemtoCells)
            self.overflown_from[timeIndex+1] = np.zeros(self.NMacroCells+self.NFemtoCells)                  # Number of users that could not be served in each BS if we had no batteries.
            self.temporal_association_vector = np.zeros(self.NMacroCells, dtype=int)
        except:
            # Last step of the simulation...
            pass
        
        self.baseStation_users[timeIndex] = np.zeros(self.NMacroCells+self.NFemtoCells)   # Number of users in each base station.
        self.active_Cells[timeIndex] = np.zeros(self.NMacroCells+self.NFemtoCells)
        self.overflown_from[timeIndex] = np.zeros(self.NMacroCells+self.NFemtoCells)      # Number of users that could not be served in each BS if we had no batteries.
        
        for userIndex in range(0, len(s_mobility['NB_USERS'])): 
            # Update position on plot of User
            self.user_pos_plot[userIndex][0].set_data([self.user_list[userIndex]["v_x"][timeIndex], self.user_list[userIndex]["v_y"][timeIndex]])

            # Search serving base station
            closestBSDownlink = simulator.map_utils.search_closest_bs([self.user_list[userIndex]["v_x"][timeIndex], self.user_list[userIndex]["v_y"][timeIndex]], self.Regions)
            
            # If closest is a Femtocell and it is sleeping (it has no users), then, check total energy consumption
            if closestBSDownlink > self.NMacroCells:
                if self.baseStation_users[timeIndex][closestBSDownlink] == 0: # If inactive
                    # Can I turn it on with PoF?
                    active_femto = np.sum(self.active_Cells[timeIndex][self.NMacroCells:])
                    current_watts = (active_femto * self.small_cell_consumption_ON) + ((self.NFemtoCells - active_femto) * self.small_cell_consumption_SLEEP)
                    if current_watts >= (self.max_energy_consumption - self.small_cell_consumption_ON + self.small_cell_consumption_SLEEP): # No, I cannot. Check battery.

                        # Check if we can use Femtocell's battery
                        if self.battery_vector[0, closestBSDownlink] > (timeStep/3600) * self.small_cell_current_draw:
                            X = [self.user_list[userIndex]["v_x"][timeIndex], self.BaseStations[closestBSDownlink, 0]]
                            Y = [self.user_list[userIndex]["v_y"][timeIndex], self.BaseStations[closestBSDownlink, 1]]
                            
                            self.user_association_line[userIndex].set_data(X, Y)
                            self.user_association_line[userIndex].set_color('green')
                            self.user_association_line[userIndex].set_linestyle('--')
                            self.user_association_line[userIndex].set_linewidth(3)
                            self.association_vector[0, userIndex] = closestBSDownlink # Associate.

                            # Alternative if we had no batteries would be...
                            self.association_vector_overflow_alternative[0, userIndex] = simulator.user_association_utils.search_closest_macro([self.user_list[userIndex]["v_x"][timeIndex], 
                                                                                                                                                self.user_list[userIndex]["v_y"][timeIndex]], 
                                                                                                                                                self.BaseStations[0:self.NMacroCells, 0:2])
                            
                            self.overflown_from[timeIndex][closestBSDownlink] += 1

                            # Legacy comment on MATLAB:
                            #active_Cells[closestBSDownlink] = 1 # This cell does not count for the overall PoF power budget.
                            self.battery_state[timeIndex][closestBSDownlink] = 2.0 # Discharge battery.
                            # However, draw from Femtocell's battery.
                            self.battery_vector[0, closestBSDownlink] = max(0, self.battery_vector[0, closestBSDownlink] - (timeStep/3600) * self.small_cell_current_draw) 
                            self.baseStation_users[timeIndex][closestBSDownlink] += 1 # Add user.
                        else:
                            # Associate to closest Macrocell
                            closest_Macro = simulator.user_association_utils.search_closest_macro([self.user_list[userIndex]["v_x"][timeIndex],
                                                                                                   self.user_list[userIndex]["v_y"][timeIndex]],
                                                                                                   self.BaseStations[0:self.NMacroCells, 0:2])
                            
                            X = [self.user_list[userIndex]["v_x"][timeIndex], self.BaseStations[closest_Macro, 0]]
                            Y = [self.user_list[userIndex]["v_y"][timeIndex], self.BaseStations[closest_Macro, 1]]

                            self.user_association_line[userIndex].set_data(X, Y)
                            self.user_association_line[userIndex].set_color('red')
                            self.user_association_line[userIndex].set_linestyle('--')
                            self.user_association_line[userIndex].set_linewidth(2)

                            self.association_vector[0, userIndex] = closest_Macro # Associate.
                            self.active_Cells[timeIndex][closest_Macro] = 1 
                            self.baseStation_users[timeIndex][closest_Macro] += 1
                    else:
                        # Yes, turn on with PoF and associate
                        X = [self.user_list[userIndex]["v_x"][timeIndex], self.BaseStations[closestBSDownlink, 0]]
                        Y = [self.user_list[userIndex]["v_y"][timeIndex], self.BaseStations[closestBSDownlink, 1]]

                        self.user_association_line[userIndex].set_data(X, Y)
                        self.user_association_line[userIndex].set_color(self.colorsBS[closestBSDownlink])
                        self.user_association_line[userIndex].set_linestyle('-')
                        self.user_association_line[userIndex].set_linewidth(0.5)

                        self.association_vector[0, userIndex] = closestBSDownlink # Associate.
                        self.association_vector_overflow_alternative[0, userIndex] = 0 # I can use PoF. Having batteries makes no difference in this case. Alternative is not needed.
                        self.active_Cells[timeIndex][closestBSDownlink] = 1 # This cell counts for the PoF budget.
                        self.battery_state[timeIndex][closestBSDownlink] = 0.0 # No battery usage.
                        self.baseStation_users[timeIndex][closestBSDownlink] += 1 # Add user.

                else: # Already ON, associate to the femtocell, just add one user.
                    self.association_vector[0, userIndex] = closestBSDownlink # Associate.

                    if self.battery_state[timeIndex][closestBSDownlink] == 2.0: # Is Discharging
                        # If we had no batteries, this user would have been gone to the closest macrocell. 
                        # Search "overflow" alternative and add 1 to the "kicked" users of this femtocell in the hypothetical case we had no batteries installed. 
                        self.association_vector_overflow_alternative[0, userIndex] = simulator.user_association_utils.search_closest_macro([self.user_list[userIndex]["v_x"][timeIndex], 
                                                                                                                                            self.user_list[userIndex]["v_y"][timeIndex]], 
                                                                                                                                            self.BaseStations[0:self.NMacroCells, 0:2])
                        self.overflown_from[timeIndex][closestBSDownlink] += 1
                    else:
                        self.association_vector_overflow_alternative[0, userIndex] = 0
                    self.baseStation_users[timeIndex][closestBSDownlink] += 1 # Add user.

                    X = [self.user_list[userIndex]["v_x"][timeIndex], self.BaseStations[closestBSDownlink, 0]]
                    Y = [self.user_list[userIndex]["v_y"][timeIndex], self.BaseStations[closestBSDownlink, 1]]

                    if self.battery_state[timeIndex][closestBSDownlink] == 2.0: # Is Discharging
                        # If using battery (only check == 2 because 3 only happens later at chaging decison)
                        self.user_association_line[userIndex].set_data(X, Y)
                        self.user_association_line[userIndex].set_color('green')
                        self.user_association_line[userIndex].set_linestyle('--')
                        self.user_association_line[userIndex].set_linewidth(3)

                    else: # Is Charging
                        self.user_association_line[userIndex].set_data(X, Y)
                        self.user_association_line[userIndex].set_color(self.colorsBS[closestBSDownlink])
                        self.user_association_line[userIndex].set_linestyle('-')
                        self.user_association_line[userIndex].set_linewidth(0.5)

            else: # Associate to a Macrocell
                X = [self.user_list[userIndex]["v_x"][timeIndex], self.BaseStations[closestBSDownlink, 0]]
                Y = [self.user_list[userIndex]["v_y"][timeIndex], self.BaseStations[closestBSDownlink, 1]]

                self.user_association_line[userIndex].set_data(X, Y)
                self.user_association_line[userIndex].set_color(self.colorsBS[closestBSDownlink])
                self.user_association_line[userIndex].set_linestyle('-')
                self.user_association_line[userIndex].set_linewidth(0.5)

                self.association_vector[0, userIndex] = closestBSDownlink # Associate.
                self.association_vector_overflow_alternative[0, userIndex] = 0                
                self.active_Cells[timeIndex][closestBSDownlink] = 1
                self.baseStation_users[timeIndex][closestBSDownlink] += 1 # Add user.
         
        # End user allocation in timeIndex instance 
        
        # Traffic calculated to user
        for userIndex in range(0, len(s_mobility['NB_USERS'])):
            self.X_user[timeIndex][userIndex][0] = self.calculate_traffic(userIndex=userIndex, timeIndex=timeIndex)
            self.X_user[timeIndex][userIndex][1] = self.calculate_traffic_no_battery(userIndex=userIndex, timeIndex=timeIndex)
            self.X_user[timeIndex][userIndex][2] = self.calculate_traffic_only_macro(userIndex=userIndex, timeIndex=timeIndex)
        
        return
    
    def compute_statistics_for_plots(self, timeIndex):
        """ Compute Statistics in order to Plot the Algorithm Output

        Args:
            timeIndex (_type_): timeSim in this moment
        """

        # Number of active Smallcells
        self.live_smallcell_occupancy[timeIndex] = np.count_nonzero(self.active_Cells[timeIndex][self.NMacroCells:])

        # Cells that overflow
        self.live_smallcell_overflow[timeIndex] = np.count_nonzero(self.overflown_from[timeIndex][self.NMacroCells:])

        # Total consumption
        self.live_smallcell_consumption[timeIndex] = (self.live_smallcell_occupancy[timeIndex] * self.small_cell_consumption_ON 
            + (self.NFemtoCells - self.live_smallcell_occupancy[timeIndex]) * self.small_cell_consumption_SLEEP)

        # Update system throughput
        self.live_throughput[timeIndex] = np.sum(self.X_macro[timeIndex]) + np.sum(self.X_femto[timeIndex])
        self.live_throughput_NO_BATTERY[timeIndex] = np.sum(self.X_macro_no_batt[timeIndex]) + np.sum(self.X_femto_no_batt[timeIndex]) + np.sum(self.X_macro_overflow[timeIndex])
        self.live_throughput_only_Macros[timeIndex] = np.sum(self.X_macro_only[timeIndex])
        return


    def update_battery_state(self, timeIndex, timeStep):
        """ Legacy function to update battery state

        Args:
            timeIndex (_type_): Index of simulation times
            timeStep (_type_): actual simulation step
        """
        # Decide about battery recharging
        if self.live_smallcell_consumption[timeIndex] < self.max_energy_consumption:
            available = self.max_energy_consumption - self.live_smallcell_consumption[timeIndex]
            I = np.argmin(self.battery_vector[0])    # TODO: why only one battery decision per timeStep?
            if self.battery_vector[0][I] < self.battery_capacity:
                charging_intensity = available / np.mean(self.small_cell_voltage_range)
                self.battery_vector[0][I] = min(self.battery_vector[0][I] + charging_intensity * (timeStep/3600), self.battery_capacity)
                try:
                    if self.battery_state[timeIndex][I] == 0.0: 
                        self.battery_state[timeIndex+1][I] = 1.0        # If none state, set as charging
                    elif self.battery_state[timeIndex][I] == 2.0: 
                        self.battery_state[timeIndex+1][I] = 3.0        # If discharging, set as charging & discharging
                    else: self.battery_state[timeIndex+1][I] = self.battery_state[timeIndex]
                except:
                    # Last step of the simulation...
                    pass
        self.battery_mean_values[timeIndex] = np.mean(self.battery_vector[0])
            
            
    def calculate_traffic(self, userIndex, timeIndex):
        """ Throughput WITH batteries given an User and timeIndex
        
        Depends of:     association_vector
                        baseStation_users <---
                        
        Returns Traffic of User
        """
        associated_station = int(self.association_vector[0][userIndex])
        SINRDLink = simulator.radio_utils.compute_sinr_dl([self.user_list[userIndex]["v_x"][timeIndex], 
                                                               self.user_list[userIndex]["v_y"][timeIndex]], 
                                                               self.BaseStations, 
                                                               associated_station, 
                                                               self.alpha_loss, 
                                                               self.PMacroCells, 
                                                               self.PFemtoCells, 
                                                               self.NMacroCells, 
                                                               self.noise)
        naturalDL = 10**(SINRDLink/10)
        if associated_station < self.NMacroCells:
            BW = self.MacroCellDownlinkBW
            X = (BW/self.baseStation_users[timeIndex][associated_station]) * np.log2(1 + naturalDL)
            self.X_macro[timeIndex][associated_station] += X
        else:
            BW = self.FemtoCellDownlinkBW
            X = (BW/self.baseStation_users[timeIndex][associated_station]) * np.log2(1 + naturalDL)
            self.X_femto[timeIndex][associated_station] += X
        return X


    def calculate_traffic_no_battery(self, userIndex, timeIndex):
        """ Throughput WITHOUT batteries given an User and timeIndex
        
        Depends of:     association_vector_overflow_alternative
                        associated_station_overflow
                        association_vector
                        baseStation_users <---
                        
        Returns Traffic of User
        """ 
        associated_station_overflow = int(self.association_vector_overflow_alternative[0][userIndex])
        if associated_station_overflow == 0:
            associated_station = int(self.association_vector[0][userIndex])
            SINRDLink = simulator.radio_utils.compute_sinr_dl([self.user_list[userIndex]["v_x"][timeIndex],
                                                               self.user_list[userIndex]["v_y"][timeIndex]],
                                                               self.BaseStations,
                                                               associated_station,
                                                               self.alpha_loss,
                                                               self.PMacroCells,
                                                               self.PFemtoCells,
                                                               self.NMacroCells,
                                                               self.noise)
            naturalDL = 10**(SINRDLink/10)
            if associated_station < self.NMacroCells:
                BW = self.MacroCellDownlinkBW
                X = (BW / (self.baseStation_users[timeIndex][associated_station] + \
                        np.sum(self.association_vector_overflow_alternative == associated_station_overflow))) * np.log2(1 + naturalDL)
                self.X_macro_no_batt[timeIndex][associated_station] += X
            else:
                BW = self.FemtoCellDownlinkBW
                # Must '+' to avoid divide by zero, in MATLAB is '-'
                X = (BW/(self.baseStation_users[timeIndex][associated_station] + \
                        self.overflown_from[timeIndex][associated_station])) * np.log2(1+naturalDL)
                self.X_femto_no_batt[timeIndex][associated_station] += X
        else:
            SINRDLink = simulator.radio_utils.compute_sinr_dl([self.user_list[userIndex]["v_x"][timeIndex],
                                                               self.user_list[userIndex]["v_y"][timeIndex]],
                                                               self.BaseStations,
                                                               associated_station_overflow,
                                                               self.alpha_loss,
                                                               self.PMacroCells,
                                                               self.PFemtoCells,
                                                               self.NMacroCells,
                                                               self.noise)
            naturalDL = 10**(SINRDLink/10)
            BW = self.MacroCellDownlinkBW
            X = (BW/(self.baseStation_users[timeIndex][int(associated_station_overflow)] + \
                    np.sum(self.association_vector_overflow_alternative[0] == associated_station_overflow))) * np.log2(1+naturalDL)
            self.X_macro_overflow[timeIndex][associated_station_overflow] += X
        return X


    def calculate_traffic_only_macro(self, userIndex, timeIndex):
        """ Throughput with ONLY Macrocells given an User and timeIndex
        
        Depends of:     nothing external
                        temporal_association_vector
                        
        Returns Traffic of User
        """
        cl = simulator.user_association_utils.search_closest_macro([self.user_list[userIndex]["v_x"][timeIndex],
                                                                        self.user_list[userIndex]["v_y"][timeIndex]],
                                                                        self.BaseStations[0:self.NMacroCells, 0:2])
        self.temporal_association_vector[cl] += 1
        SINRDLink = simulator.radio_utils.compute_sinr_dl([self.user_list[userIndex]["v_x"][timeIndex],
                                                           self.user_list[userIndex]["v_y"][timeIndex]],
                                                           self.BaseStations,
                                                           cl,
                                                           self.alpha_loss,
                                                           self.PMacroCells,
                                                           self.PFemtoCells,
                                                           self.NMacroCells,
                                                           self.noise)
        naturalDL = 10**(SINRDLink/10)
        BW = self.MacroCellDownlinkBW
        X = (BW / self.temporal_association_vector[cl]) * np.log2(1 + naturalDL)
        self.X_macro_only[timeIndex][cl] += X
        return X
    
    def plot_output(self, sim_times, show_plots: bool = True):
        """ Override Show Plot Output

        Args:
            sim_times (_type_): _description_
            show_plots (bool, optional): _description_. Defaults to True.
        """
        # User Traffic 
        fig_user_traffic, ax = plt.subplots()
        self.list_figures.append((fig_user_traffic, "user-traffic"))    # In Order to save the figure on output folder
        userIndex = 10
        metric = 0  # Default traffic
        user_traffic = np.asarray([self.X_user[t][userIndex][metric] for t in range(len(sim_times))])
        ax.plot(sim_times, user_traffic/10e6, label='With battery system')
        ax.legend()
        ax.set_title(f'User {userIndex} Traffic')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Throughput [Mb/s]')
        
        # Batteries in use for each timeStep
        battery_charging = []
        for timeIndex in self.battery_state:
            # battery charging == 1 or 3
            count_3 = np.count_nonzero(timeIndex == 3.0)
            count_1 = np.count_nonzero(timeIndex == 1.0)
            battery_charging.append(count_3 + count_1)
            
        fig_battery_charging, ax = plt.subplots()
        self.list_figures.append((fig_battery_charging, "charging-cells"))    # In Order to save the figure on output folder
        ax.plot(sim_times, battery_charging, label="Charging Cells")
        ax.legend()
        ax.set_title("Charging Cells")
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Number of cells')
        
        # Get the context_class method
        super().plot_output(sim_times=sim_times, show_plots=show_plots)
