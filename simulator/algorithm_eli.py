import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

from simulator.launch import logger
from simulator.context_config import Contex_Config
import simulator.map_utils, simulator.mobility_utils, simulator.user_association_utils, simulator.radio_utils

class PoF_simulation_ELi(Contex_Config):
    def start_simulation(self, sim_times, timeStep, s_mobility, text_plot, show_plots: bool = True, speed_plot: float = 0.05):
        self.battery_state = [[] for _ in range(len(sim_times))]
        self.baseStation_users = [[] for _ in range(len(sim_times))]
        self.active_Cells = [[] for _ in range(len(sim_times))]
        self.overflown_from = [[] for _ in range(len(sim_times))]
        
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
                self.compute_statistics(timeIndex=timeIndex, s_mobility=s_mobility)
                self.update_battery_status(timeIndex=timeIndex, timeStep=timeStep)
                
                if show_plots:
                    plt.draw()
                    plt.pause(speed_plot)
                
        # Finished
        logger.info("Simulation complete!")
        logger.info(f"Elapsed time: {np.round(time.time() - start, decimals=4)} seconds.")
        return
        
    def algorithm_step(self, timeIndex, timeStep, s_mobility):
        try:
            self.battery_state[timeIndex+1] = np.zeros(self.NMacroCells+self.NFemtoCells)       # 0 = nothing; 1 = charging; 2 = discharging; 3 = discharging & charging.
            self.baseStation_users[timeIndex+1] = np.zeros(self.NMacroCells+self.NFemtoCells)   # Number of users in each base station.
            self.active_Cells[timeIndex+1] = np.zeros(self.NMacroCells+self.NFemtoCells)
            self.overflown_from[timeIndex+1] = np.zeros(self.NMacroCells+self.NFemtoCells)      # Number of users that could not be served in each BS if we had no batteries.
        except:
            # Last step of the simulation...
            pass
        
        self.battery_state[timeIndex] = np.zeros(self.NMacroCells+self.NFemtoCells)       # 0 = nothing; 1 = charging; 2 = discharging; 3 = discharging & charging.
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
                if self.baseStation_users[timeIndex][closestBSDownlink] == 0: #If inactive
                    #Can I turn it on with PoF?
                    active_femto = np.sum(self.active_Cells[timeIndex][self.NMacroCells:])
                    current_watts = (active_femto * self.small_cell_consumption_ON) + ((self.NFemtoCells - active_femto) * self.small_cell_consumption_SLEEP)
                    if current_watts >= (self.max_energy_consumption - self.small_cell_consumption_ON + self.small_cell_consumption_SLEEP): # No, I cannot. Check battery.

                        #Check if we can use Femtocell's battery
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

                            # Comment on MATLAB:
                            #active_Cells[closestBSDownlink] = 1 # This cell does not count for the overall PoF power budget.
                            self.battery_state[timeIndex][closestBSDownlink] = 2 # Discharge battery.
                            # However, draw from Femtocell's battery.
                            self.battery_vector[0, closestBSDownlink] = max(0, self.battery_vector[0, closestBSDownlink] - (timeStep/3600) * self.small_cell_current_draw) 
                            self.baseStation_users[timeIndex][closestBSDownlink] += 1 # Add user.
                        else:
                            #Associate to closest Macrocell
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
                        #Yes, turn on with PoF and associate
                        X = [self.user_list[userIndex]["v_x"][timeIndex], self.BaseStations[closestBSDownlink, 0]]
                        Y = [self.user_list[userIndex]["v_y"][timeIndex], self.BaseStations[closestBSDownlink, 1]]

                        self.user_association_line[userIndex].set_data(X, Y)
                        self.user_association_line[userIndex].set_color(self.colorsBS[closestBSDownlink])
                        self.user_association_line[userIndex].set_linestyle('-')
                        self.user_association_line[userIndex].set_linewidth(0.5)

                        self.association_vector[0, userIndex] = closestBSDownlink # Associate.
                        self.association_vector_overflow_alternative[0, userIndex] = 0 # I can use PoF. Having batteries makes no difference in this case. Alternative is not needed.
                        self.active_Cells[timeIndex][closestBSDownlink] = 1 # This cell counts for the PoF budget.
                        self.battery_state[timeIndex][closestBSDownlink] = 0 # No battery usage.
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

                    else:   # Is Charging
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
                
        return
    
    def compute_statistics(self, timeIndex, s_mobility):
        # Throughput WITH batteries
        total_DL_Throughput = 0
        for userIndex in range(0, len(s_mobility['NB_USERS'])):
            SINRDLink = simulator.radio_utils.compute_sinr_dl([self.user_list[userIndex]["v_x"][timeIndex], 
                                                               self.user_list[userIndex]["v_y"][timeIndex]], 
                                                               self.BaseStations, 
                                                               self.association_vector[0][userIndex], 
                                                               self.alpha_loss, 
                                                               self.PMacroCells, 
                                                               self.PFemtoCells, 
                                                               self.NMacroCells, 
                                                               self.noise)
            naturalDL = 10**(SINRDLink/10)
            if self.association_vector[0][userIndex] < self.NMacroCells:
                BW = self.MacroCellDownlinkBW
            else:
                BW = self.FemtoCellDownlinkBW
            RateDL = (BW/self.baseStation_users[timeIndex][int(self.association_vector[0][userIndex])]) * np.log2(1 + naturalDL)
            total_DL_Throughput += RateDL

        # Throughput WITHOUT batteries
        total_DL_Throughput_overflow_alternative = 0
        for userIndex in range(0, len(s_mobility['NB_USERS'])):
            if self.association_vector_overflow_alternative[0][userIndex] == 0.0:
                SINRDLink = simulator.radio_utils.compute_sinr_dl([self.user_list[userIndex]["v_x"][timeIndex],
                                                                   self.user_list[userIndex]["v_y"][timeIndex]],
                                                                   self.BaseStations,
                                                                   self.association_vector[0][userIndex],
                                                                   self.alpha_loss,
                                                                   self.PMacroCells,
                                                                   self.PFemtoCells,
                                                                   self.NMacroCells,
                                                                   self.noise)
                naturalDL = 10**(SINRDLink/10)
                if self.association_vector[0][userIndex] < self.NMacroCells:
                    BW = self.MacroCellDownlinkBW
                    RateDL = (BW / (self.baseStation_users[timeIndex][int(self.association_vector[0][userIndex])] + \
                                    np.sum(self.association_vector_overflow_alternative == self.association_vector_overflow_alternative[0][userIndex]))) * np.log2(1 + naturalDL)
                else:
                    BW = self.FemtoCellDownlinkBW
                    # Must '+' to avoid divide by zero, in MATLAB is '-'
                    RateDL = (BW/(self.baseStation_users[timeIndex][int(self.association_vector[0][userIndex])] + self.overflown_from[timeIndex][int(self.association_vector[0][userIndex])])) * np.log2(1+naturalDL)
                total_DL_Throughput_overflow_alternative += RateDL 
            else:
                SINRDLink = simulator.radio_utils.compute_sinr_dl([self.user_list[userIndex]["v_x"][timeIndex],
                                                                   self.user_list[userIndex]["v_y"][timeIndex]],
                                                                   self.BaseStations,
                                                                   self.association_vector_overflow_alternative[0][userIndex],
                                                                   self.alpha_loss,
                                                                   self.PMacroCells,
                                                                   self.PFemtoCells,
                                                                   self.NMacroCells,
                                                                   self.noise)
                naturalDL = 10**(SINRDLink/10)
                BW = self.MacroCellDownlinkBW
                RateDL = (BW/(self.baseStation_users[timeIndex][int(self.association_vector_overflow_alternative[0][userIndex])] + \
                    np.sum(self.association_vector_overflow_alternative[0] == self.association_vector_overflow_alternative[0][userIndex]))) * np.log2(1+naturalDL)
                total_DL_Throughput_overflow_alternative += RateDL

        # Throughput with ONLY Macrocells
        total_DL_Throughput_only_Macros = 0
        temporal_association_vector = np.zeros(self.NMacroCells, dtype=int)
        for userIndex in range(0, len(s_mobility['NB_USERS'])):
            cl = simulator.user_association_utils.search_closest_macro([self.user_list[userIndex]["v_x"][timeIndex],
                                                                        self.user_list[userIndex]["v_y"][timeIndex]],
                                                                        self.BaseStations[0:self.NMacroCells, 0:2])
            temporal_association_vector[cl] += 1
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
            RateDL = (BW / temporal_association_vector[cl]) * np.log2(1 + naturalDL)
            total_DL_Throughput_only_Macros += RateDL

        # Compute the number of active Smallcells
        self.live_smallcell_occupancy[timeIndex] = np.count_nonzero(self.active_Cells[timeIndex][self.NMacroCells:])

        # Compute the cells that overflow
        self.live_smallcell_overflow[timeIndex] = np.count_nonzero(self.overflown_from[timeIndex][self.NMacroCells:])

        # Compute the total consumption
        self.live_smallcell_consumption[timeIndex] = (self.live_smallcell_occupancy[timeIndex] * self.small_cell_consumption_ON 
            + (self.NFemtoCells - self.live_smallcell_occupancy[timeIndex]) * self.small_cell_consumption_SLEEP)

        # Update system throughput
        self.live_throughput[timeIndex] = total_DL_Throughput
        self.live_throughput_NO_BATTERY[timeIndex] = total_DL_Throughput_overflow_alternative
        self.live_throughput_only_Macros[timeIndex] = total_DL_Throughput_only_Macros
        
        return

    def update_battery_status(self, timeIndex, timeStep):
        # Decide about battery recharging
            if self.live_smallcell_consumption[timeIndex] < self.max_energy_consumption:
                available = self.max_energy_consumption - self.live_smallcell_consumption[timeIndex]
                I = np.argmin(self.battery_vector[0])    # TODO: why only one battery decision per timeStep?
                if self.battery_vector[0][I] < self.battery_capacity:
                    charging_intensity = available / np.mean(self.small_cell_voltage_range)
                    self.battery_vector[0][I] = min(self.battery_vector[0][I] + charging_intensity * (timeStep/3600), self.battery_capacity)
                    try:
                        if self.battery_state[timeIndex][I] == 0.0: self.battery_state[timeIndex+1][I] = 1.0      # If none state, set as charging
                        elif self.battery_state[timeIndex][I] == 2.0: self.battery_state[timeIndex+1][I] = 3.0    # If discharging, set as charging & discharging
                        else: self.battery_state[timeIndex+1][I] = self.battery_state[timeIndex]
                    except:
                        # Last step of the simulation...
                        pass
            self.battery_mean_values[timeIndex] = np.mean(self.battery_vector[0])

