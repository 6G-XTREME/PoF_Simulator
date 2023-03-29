import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import time, random

from simulator.launch import logger
from simulator.context_config import Contex_Config
import simulator.map_utils, simulator.mobility_utils, simulator.user_association_utils, simulator.radio_utils

class PoF_simulation_ELighthouse(Contex_Config):
    user_report_position: int                   # TimeStep between user new position report
    user_report_position_next_time: np.array    # Next TimeStep to report position (with random)
    user_closest_bs: np.array                   # To Save previous closestBS
    startup_max_tokens: int                     # Max tokens to count down to BS count as active
    poweroff_max_tokens: int                    # TimeSteps to poweroff a cell
    
    starting_up_femto: np.array     # Save the token state
    started_up_femto: np.array      # Array of FemtoCell ON
    
    is_in_femto: np.array                   # Save if user is in femto area, served or not
    timeIndex_first_battery_dead: int       
    dead_batteries: list                    # Save the batteries that already died
    timeIndex_last_battery_dead: int
    
    # Traffic Vars
    X_macro_bps: np.array
    X_macro_only_bps: np.array
    X_macro_no_batt_bps: np.array
    X_macro_overflow_bps: np.array
    X_femto_bps: np.array
    X_femto_no_batt_bps: np.array
    X_user_bps: np.array
    
    output_throughput: np.array
    output_throughput_no_batt: np.array
    output_throughput_only_macro: np.array
    
    def __init__(self, sim_times, basestation_data: dict, user_data: dict, battery_data: dict, transmit_power_data: dict, elighthouse_parameters: dict) -> None:
        # Set seed for random
        random.seed(150)
        
        try:
            # Number of timeSlots that user should wait to re-send the position
            if elighthouse_parameters['user_report_position'] > 0 and elighthouse_parameters['user_report_position'] < 20:
                self.user_report_position = elighthouse_parameters['user_report_position']
            else:
                self.user_report_position = 1   # For each timeStep, the user report his position
            
            # Number of timeSlots to startup a femtocell 
            if elighthouse_parameters['startup_max_tokens'] > 0 and elighthouse_parameters['startup_max_tokens'] < 10:
                self.startup_max_tokens = elighthouse_parameters['startup_max_tokens']
            else:
                self.startup_max_tokens = 1
               
            # Number of timeSlots to Poweroff a non used Cell
            if elighthouse_parameters['poweroff_unused_cell'] > 0 and elighthouse_parameters['poweroff_unused_cell'] < 10:
                self.poweroff_max_tokens = elighthouse_parameters['poweroff_unused_cell']
            else:
                self.poweroff_max_tokens = 1
        except:
            # On error, load default custom parameters
            self.user_report_position = 1
            self.startup_max_tokens = 1
            self.poweroff_max_tokens = 1
        
        super().__init__(sim_times=sim_times, basestation_data=basestation_data, user_data=user_data, battery_data=battery_data, transmit_power_data=transmit_power_data)
    
    def start_simulation(self, sim_times, timeStep, text_plot, show_plots: bool = True, speed_plot: float = 0.05):
        # Settting up some vars
        self.battery_state = [[] for _ in range(len(sim_times))]
        self.baseStation_users = [[] for _ in range(len(sim_times))]
        self.active_Cells = [[] for _ in range(len(sim_times))]
        self.overflown_from = [[] for _ in range(len(sim_times))]
        
        self.user_report_position_next_time = [self.user_report_position for _ in range(len(self.NUsers))]
        self.user_closest_bs = np.zeros((len(sim_times), len(self.NUsers)))
        self.starting_up_femto = np.zeros(self.NMacroCells + self.NFemtoCells)
        self.started_up_femto = []
        
        self.is_in_femto = np.zeros((len(self.NUsers), len(sim_times)), dtype=int)
        self.timeIndex_first_battery_dead = 0
        self.dead_batteries = []
        self.timeIndex_last_battery_dead = 0
        
        # Traffic global vars
        self.X_macro_bps = np.zeros((len(sim_times), self.NMacroCells))
        self.X_macro_only_bps = np.zeros((len(sim_times), self.NMacroCells))
        self.X_macro_no_batt_bps = np.zeros((len(sim_times), self.NMacroCells))
        self.X_macro_overflow_bps = np.zeros((len(sim_times), self.NMacroCells))
        self.X_femto_bps = np.zeros((len(sim_times), self.NMacroCells+self.NFemtoCells))
        self.X_femto_no_batt_bps = np.zeros((len(sim_times), self.NMacroCells+self.NFemtoCells))
        
        # Three metrics to save: 
        # 0. Global system with batteries
        # 1. Femto without batteries
        # 2. Only Macro
        metrics = 3
        self.X_user = np.zeros((len(sim_times), len(self.NUsers), metrics), dtype=float)
        
        # Init new plot vars
        self.output_throughput = np.zeros((2, len(sim_times)))          # 0: macro, 1: femto
        self.output_throughput_no_batt = np.zeros((3, len(sim_times)))    # 0: macro 1: femto 2: overflow
        self.output_throughput_only_macro = np.zeros(len(sim_times))    # 0: macro
        
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

                self.algorithm_step(timeIndex=timeIndex, timeStep=timeStep)
                self.compute_statistics_for_plots(timeIndex=timeIndex)                          # Prepare derivate data for plots
                self.update_battery_state(timeIndex=timeIndex, timeStep=timeStep)               # Update battery state for next timeStep
                
                if show_plots:
                    plt.draw()
                    plt.pause(speed_plot)
                
        # Finished
        logger.info("Simulation complete!")
        logger.info(f"Elapsed time: {np.round(time.time() - start, decimals=4)} seconds.")
        return

 
    def algorithm_step(self, timeIndex, timeStep):
        """ Algorithm Logic to execute in each timeStep of the simulation

        Args:
            timeIndex (int): 
            timeStep (float): 
        """
        if timeIndex == 0: self.battery_state[timeIndex] = np.zeros(self.NMacroCells+self.NFemtoCells)      # 0 = nothing; 1 = charging; 2 = discharging; 3 = discharging & charging.
        if self.started_up_femto is None: self.started_up_femto = []                                        # The case that, all the started femto go to off
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
        
        for userIndex in range(0, len(self.NUsers)): 
            # Update position on plot of User
            self.user_pos_plot[userIndex][0].set_data([self.user_list[userIndex]["v_x"][timeIndex], self.user_list[userIndex]["v_y"][timeIndex]])

            # Search serving base station
            if (timeIndex == 0 or timeIndex == self.user_report_position_next_time[userIndex]):
                # New Position Report
                closestBSDownlink = simulator.map_utils.search_closest_bs([self.user_list[userIndex]["v_x"][timeIndex], self.user_list[userIndex]["v_y"][timeIndex]], self.Regions)
                if (timeIndex != 0): self.user_report_position_next_time[userIndex] += random.randint(1, self.user_report_position+1)
            else:
                # Use previous position know
                closestBSDownlink = int(self.user_closest_bs[timeIndex-1][userIndex])
            # Update the actual BS
            self.user_closest_bs[timeIndex][userIndex] = closestBSDownlink
            
            # If closest is a Femtocell and it is sleeping (it has no users), then, check total energy consumption
            if closestBSDownlink > self.NMacroCells:
                if self.baseStation_users[timeIndex][closestBSDownlink] == 0: # If inactive
                    # Can I turn it on with PoF?
                    active_femto = np.sum(self.active_Cells[timeIndex][self.NMacroCells:])
                    battery_femto = np.count_nonzero(self.battery_state[timeIndex][self.NMacroCells:] == 2.0)   # Battery Cells doesnt count for current_watts budget
                    current_watts = (active_femto * self.small_cell_consumption_ON) + ((self.NFemtoCells - (active_femto + battery_femto)) * self.small_cell_consumption_SLEEP)
                    if current_watts >= (self.max_energy_consumption - self.small_cell_consumption_ON + self.small_cell_consumption_SLEEP): # No, I cannot. Check battery.

                        # Check if we can use Femtocell's battery
                        if self.battery_vector[0, closestBSDownlink] > (timeStep/3600) * self.small_cell_current_draw:
                            
                            # Check if is booting Up!
                            if (self.starting_up_femto[closestBSDownlink] > 0 and self.starting_up_femto[closestBSDownlink] <= self.startup_max_tokens):
                                # On Process to startup, using backup...
                                self.active_Cells[timeIndex][closestBSDownlink] = 0     # This cell does not count for the overall PoF power budget.
                                self.battery_state[timeIndex][closestBSDownlink] = 2.0  # Discharge battery.
                            # Check if already booted
                            elif closestBSDownlink in self.started_up_femto:
                                # Yes! Femto is booted
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

                                self.is_in_femto[userIndex][timeIndex] = 1              # User is associated with femto
                                self.active_Cells[timeIndex][closestBSDownlink] = 0     # This cell does not count for the overall PoF power budget.
                                self.battery_state[timeIndex][closestBSDownlink] = 2.0  # Discharge battery.
                                # However, draw from Femtocell's battery.
                                self.battery_vector[0, closestBSDownlink] = max(0, self.battery_vector[0, closestBSDownlink] - (timeStep/3600) * self.small_cell_current_draw) 
                                self.baseStation_users[timeIndex][closestBSDownlink] += 1 # Add user.
                                continue
                            else:
                                # Not started, should startup
                                self.starting_up_femto[closestBSDownlink] = self.startup_max_tokens       # Init Bucket for this closest BS
                        
                                self.active_Cells[timeIndex][closestBSDownlink] = 0     # This cell does not count for the overall PoF power budget.
                                self.battery_state[timeIndex][closestBSDownlink] = 2.0  # Discharge battery.
                            
                            # Associate User to closest Macro
                            closest_Macro = simulator.user_association_utils.search_closest_macro([self.user_list[userIndex]["v_x"][timeIndex],
                                                                                                    self.user_list[userIndex]["v_y"][timeIndex]],
                                                                                                    self.BaseStations[0:self.NMacroCells, 0:2])
                        
                            X = [self.user_list[userIndex]["v_x"][timeIndex], self.BaseStations[closest_Macro, 0]]
                            Y = [self.user_list[userIndex]["v_y"][timeIndex], self.BaseStations[closest_Macro, 1]]
                        
                            self.user_association_line[userIndex].set_data(X, Y)
                            self.user_association_line[userIndex].set_color('orange')
                            self.user_association_line[userIndex].set_linestyle('--')
                            self.user_association_line[userIndex].set_linewidth(0.5)
                        
                            self.association_vector[0, userIndex] = closest_Macro   # Associate.
                            self.is_in_femto[userIndex][timeIndex] = 2              # In Femto area, but associate with macro
                            self.active_Cells[timeIndex][closest_Macro] = 1 
                            self.baseStation_users[timeIndex][closest_Macro] += 1
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
                            self.is_in_femto[userIndex][timeIndex] = 2              # In Femto area, but associate with macro
                            self.active_Cells[timeIndex][closest_Macro] = 1 
                            self.baseStation_users[timeIndex][closest_Macro] += 1
                    else:
                        # Yes, turn on with PoF and try to associate
                        
                        # Check if BS is already on or not started
                        if (self.starting_up_femto[closestBSDownlink] > 0 and self.starting_up_femto[closestBSDownlink] <= self.startup_max_tokens):
                            # On Process to startup, using backup...
                            self.active_Cells[timeIndex][closestBSDownlink] = 1     # This cell counts for the PoF budget.
                            self.battery_state[timeIndex][closestBSDownlink] = 0.0  # No battery usage.
                        elif closestBSDownlink in self.started_up_femto:
                            # Femto on! Yes, associate
                            X = [self.user_list[userIndex]["v_x"][timeIndex], self.BaseStations[closestBSDownlink, 0]]
                            Y = [self.user_list[userIndex]["v_y"][timeIndex], self.BaseStations[closestBSDownlink, 1]]

                            self.user_association_line[userIndex].set_data(X, Y)
                            self.user_association_line[userIndex].set_color(self.colorsBS[closestBSDownlink])
                            self.user_association_line[userIndex].set_linestyle('-')
                            self.user_association_line[userIndex].set_linewidth(0.5)

                            self.association_vector[0, userIndex] = closestBSDownlink       # Associate.
                            self.association_vector_overflow_alternative[0, userIndex] = 0  # I can use PoF. Having batteries makes no difference in this case. Alternative is not needed.
                            self.is_in_femto[userIndex][timeIndex] = 1                      # User is associated with femto
                            self.active_Cells[timeIndex][closestBSDownlink] = 1             # This cell counts for the PoF budget.
                            self.battery_state[timeIndex][closestBSDownlink] = 0.0          # No battery usage.
                            self.baseStation_users[timeIndex][closestBSDownlink] += 1       # Add user.
                            continue 
                        else:
                            # Not started, should startup
                            self.starting_up_femto[closestBSDownlink] = self.startup_max_tokens       # Init Bucket for this closest BS
                        
                            self.active_Cells[timeIndex][closestBSDownlink] = 1 # This cell counts for the PoF budget.
                            self.battery_state[timeIndex][closestBSDownlink] = 0.0 # No battery usage.
                            
                        # Associate User to closest Macro
                        closest_Macro = simulator.user_association_utils.search_closest_macro([self.user_list[userIndex]["v_x"][timeIndex],
                                                                                               self.user_list[userIndex]["v_y"][timeIndex]],
                                                                                               self.BaseStations[0:self.NMacroCells, 0:2])
                        
                        X = [self.user_list[userIndex]["v_x"][timeIndex], self.BaseStations[closest_Macro, 0]]
                        Y = [self.user_list[userIndex]["v_y"][timeIndex], self.BaseStations[closest_Macro, 1]]
                        
                        self.user_association_line[userIndex].set_data(X, Y)
                        self.user_association_line[userIndex].set_color('orange')
                        self.user_association_line[userIndex].set_linestyle('--')
                        self.user_association_line[userIndex].set_linewidth(0.5)
                        
                        self.association_vector[0, userIndex] = closest_Macro   # Associate.
                        self.is_in_femto[userIndex][timeIndex] = 2              # In Femto area, but associate with macro
                        self.active_Cells[timeIndex][closest_Macro] = 1 
                        self.baseStation_users[timeIndex][closest_Macro] += 1

                else: # Already ON, associate to the femtocell, just add one user.
                    # Check if femto cell is booting up... (token bucket its zero if already booted)
                    if self.starting_up_femto[closestBSDownlink] == 0:
                        self.association_vector[0, userIndex] = closestBSDownlink   # Associate.
                        self.is_in_femto[userIndex][timeIndex] = 1                  # User in Femto area and associated with that cell

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
                    else:
                        # Token bucket isnt zero, so still booting up
                        # Associate user with closest MacroCell
                        closest_Macro = simulator.user_association_utils.search_closest_macro([self.user_list[userIndex]["v_x"][timeIndex],
                                                                                               self.user_list[userIndex]["v_y"][timeIndex]],
                                                                                               self.BaseStations[0:self.NMacroCells, 0:2])
                        
                        X = [self.user_list[userIndex]["v_x"][timeIndex], self.BaseStations[closest_Macro, 0]]
                        Y = [self.user_list[userIndex]["v_y"][timeIndex], self.BaseStations[closest_Macro, 1]]
                        
                        self.user_association_line[userIndex].set_data(X, Y)
                        self.user_association_line[userIndex].set_color('orange')
                        self.user_association_line[userIndex].set_linestyle('--')
                        self.user_association_line[userIndex].set_linewidth(0.5)
                        
                        self.association_vector[0, userIndex] = closest_Macro   # Associate.
                        self.is_in_femto[userIndex][timeIndex] = 2              # User in femto area, but not associated
                        self.active_Cells[timeIndex][closest_Macro] = 1 
                        self.baseStation_users[timeIndex][closest_Macro] += 1

            else: # Associate to a Macrocell
                X = [self.user_list[userIndex]["v_x"][timeIndex], self.BaseStations[closestBSDownlink, 0]]
                Y = [self.user_list[userIndex]["v_y"][timeIndex], self.BaseStations[closestBSDownlink, 1]]

                self.user_association_line[userIndex].set_data(X, Y)
                self.user_association_line[userIndex].set_color(self.colorsBS[closestBSDownlink])
                self.user_association_line[userIndex].set_linestyle('-')
                self.user_association_line[userIndex].set_linewidth(0.5)

                self.association_vector[0, userIndex] = closestBSDownlink # Associate.
                self.association_vector_overflow_alternative[0, userIndex] = 0                
                self.active_Cells[timeIndex][closestBSDownlink] = 0         # User not in area of femtocell
                self.baseStation_users[timeIndex][closestBSDownlink] += 1   # Add user.
         
        # End user allocation in timeIndex instance 
        
        # Given the list of started femto, check if have user already, if not, shutdown
        if timeIndex % self.poweroff_max_tokens == 0:   # Each two cycles, poweroff cells
            for femto in range(0, len(self.started_up_femto)):
                try:
                    if self.baseStation_users[timeIndex][self.started_up_femto[femto]] == 0:
                        # No user found in this timeIndex
                        # Powering off the femto
                        self.started_up_femto.remove(self.started_up_femto[femto])
                except: 
                    # Startep up femto is empty, zero cells are on
                    pass
        
        # Reduce the token bucket for starting up a BaseStation
        for femto in range(0,len(self.starting_up_femto)):
            if self.starting_up_femto[femto] > 0:
                # Reduce Token Bucket
                if self.starting_up_femto[femto] == 1:
                    # Add to started femto
                    try:
                        self.started_up_femto.append(femto)    
                    except:
                        self.started_up_femto = []
                        self.started_up_femto.append(femto)
                # Reduce token
                self.starting_up_femto[femto] = self.starting_up_femto[femto] - 1    
        
        # Traffic calculated to user
        for userIndex in range(0, len(self.NUsers)):
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
        self.output_throughput[0][timeIndex] = np.sum(self.X_macro_bps[timeIndex])
        self.output_throughput[1][timeIndex] = np.sum(self.X_femto_bps[timeIndex])
        self.output_throughput_no_batt[0][timeIndex] = np.sum(self.X_macro_no_batt_bps[timeIndex])
        self.output_throughput_no_batt[1][timeIndex] = np.sum(self.X_femto_no_batt_bps[timeIndex])
        self.output_throughput_no_batt[2][timeIndex] = np.sum(self.X_macro_overflow_bps[timeIndex])
        self.output_throughput_only_macro[timeIndex] = np.sum(self.X_macro_only_bps[timeIndex])
        
        self.live_throughput[timeIndex] = np.sum(self.X_macro_bps[timeIndex]) + np.sum(self.X_femto_bps[timeIndex])
        self.live_throughput_NO_BATTERY[timeIndex] = np.sum(self.X_macro_no_batt_bps[timeIndex]) + np.sum(self.X_femto_no_batt_bps[timeIndex]) + np.sum(self.X_macro_overflow_bps[timeIndex]) 
        self.live_throughput_only_Macros[timeIndex] = np.sum(self.X_macro_only_bps[timeIndex])
        return


    def update_battery_state(self, timeIndex, timeStep):
        """ Legacy function to update battery state

        Args:
            timeIndex (_type_): Index of simulation times
            timeStep (_type_): actual simulation step
        """
        # Decide about battery recharging
        if self.live_smallcell_consumption[timeIndex] < self.max_energy_consumption:
            # Asign available energy to charge a cell battery
            available = self.max_energy_consumption - self.live_smallcell_consumption[timeIndex]
            I = np.argmin(self.battery_vector[0])    # One decision for eachTimeStep -> Because we can concentrate the laser power
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
        
        # Check if battery is dead [Only femtocells]
        for batt in range(self.NMacroCells, len(self.battery_vector[0])):
            battery_capacity = round(self.battery_vector[0][batt], 2)
            if battery_capacity == 0:
                if self.timeIndex_first_battery_dead == 0:
                    # First battery dead
                    self.timeIndex_first_battery_dead = timeIndex
                    self.dead_batteries.append(batt)
                else:
                    # Already found the first battery_dead
                    if not batt in self.dead_batteries:
                        self.timeIndex_last_battery_dead = timeIndex
                        self.dead_batteries.append(batt)
            
    def calculate_traffic(self, userIndex, timeIndex):
        """ Throughput WITH batteries given an User and timeIndex
        
        Depends of:     association_vector
                        baseStation_users
                        
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
            self.X_macro_bps[timeIndex][associated_station] += X
        else:
            BW = self.FemtoCellDownlinkBW
            X = (BW/self.baseStation_users[timeIndex][associated_station]) * np.log2(1 + naturalDL)
            self.X_femto_bps[timeIndex][associated_station] += X
        return X


    def calculate_traffic_no_battery(self, userIndex, timeIndex):
        """ Throughput WITHOUT batteries given an User and timeIndex
        
        Depends of:     association_vector_overflow_alternative
                        associated_station_overflow
                        association_vector
                        baseStation_users
                        
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
                self.X_macro_no_batt_bps[timeIndex][associated_station] += X
            else:
                BW = self.FemtoCellDownlinkBW
                # Must '+' to avoid divide by zero, in MATLAB is '-'
                X = (BW/(self.baseStation_users[timeIndex][associated_station] + \
                        self.overflown_from[timeIndex][associated_station])) * np.log2(1+naturalDL)
                self.X_femto_no_batt_bps[timeIndex][associated_station] += X
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
            self.X_macro_overflow_bps[timeIndex][associated_station_overflow] += X
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
        self.X_macro_only_bps[timeIndex][cl] += X
        return X
    
    def plot_output(self, sim_times, timeStep, show_plots: bool = True):
        """ Override Show Plot Output

        Args:
            sim_times (_type_): _description_
            show_plots (bool, optional): _description_. Defaults to True.
        """
        # Battery dead
        if self.timeIndex_first_battery_dead != 0:
            # First
            print(f"First Battery dead at timeIndex: {self.timeIndex_first_battery_dead} ({(self.timeIndex_first_battery_dead*timeStep)/60} min)")
            # Last
            print(f"Last Battery dead at timeIndex: {self.timeIndex_last_battery_dead} ({(self.timeIndex_last_battery_dead*timeStep)/60} min)")
        
        # TODO: What user? Aggregated? Mean? Ask Javier
        # Print % time in femto area for X user
        user = 2 
        # Served
        per_served_associated =  round((np.count_nonzero(self.is_in_femto[user] == 1) / len(self.is_in_femto[user])) * 100, 3)
        print(f"Porcentaje de tiempo en el area de una femto y asociado a ella: {per_served_associated}")
        time_in_femto =  round(((np.count_nonzero(self.is_in_femto[user] == 1) + np.count_nonzero(self.is_in_femto[user] == 2))  / len(self.is_in_femto[user])) * 100, 3)
        print(f"Porcentaje de tiempo dentro del area de la femto: {time_in_femto}")
        print(f"Porcentaje de tiempo dentro del area de la femto pero asociado a una macro: {time_in_femto - per_served_associated}")
        
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
        for timeIndex in self.battery_state:
            # 0 = nothing; 1 = charging; 2 = discharging; 3 = discharging & charging.
            #count_3 = np.count_nonzero(timeIndex == 3.0)
            #count_1 = np.count_nonzero(timeIndex == 1.0)
            count_2 = np.count_nonzero(timeIndex == 2.0)
            #battery_charging.append(count_3 + count_1)
            battery_charging.append(count_2)
            
        fig_battery_charging, ax = plt.subplots()
        self.list_figures.append((fig_battery_charging, "discharging-cells"))    # In Order to save the figure on output folder
        ax.plot(sim_times, battery_charging, label="Discharging Cells")
        ax.legend()
        ax.set_title("Discharging Battery Cells")
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Number of cells')
        
        # Battery capacity
        fig_batt_capacity, ax = plt.subplots()
        self.list_figures.append((fig_batt_capacity, "batt-capacity")) 
        ax.axhline(y=3300, color='r',label="Max. capacity")
        for bar in range(0, len(self.battery_vector[0])):
            if bar >= self.NMacroCells:
                ax.bar(int(bar), self.battery_vector[0][bar]*1000)
        ax.legend()
        ax.set_title("Battery Capacity")
        ax.set_xlabel("Femto cell number")
        ax.set_ylabel("Capacity [mAh]")
        
        # New Figures
        ## Throughput
        fig_throughput, ax = plt.subplots()
        self.list_figures.append((fig_throughput, 'output-throughput'))
        ax.plot(sim_times, self.output_throughput[0]/10e6, label="Macro Cells")
        ax.plot(sim_times, self.output_throughput[1]/10e6, label="Femto Cells")
        ax.plot(sim_times, self.live_throughput/10e6, label="Total")
        ax.legend()
        ax.set_title("Throughput Downlink. System with batteries")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel('Throughput [Mb/s]')
        
        ## Throughput no battery
        fig_throughput_no_batt, ax = plt.subplots()
        self.list_figures.append((fig_throughput_no_batt, 'output-throughput-no-batt'))
        ax.plot(sim_times, self.output_throughput_no_batt[0]/10e6, label="Macro Cells")
        ax.plot(sim_times, self.output_throughput_no_batt[1]/10e6, label="Femto Cells")
        ax.plot(sim_times, self.output_throughput_no_batt[2]/10e6, label="Femto Cells overflow")
        ax.plot(sim_times, self.live_throughput_NO_BATTERY/10e6, label="Total")
        ax.legend()
        ax.set_title("Throughput Downlink. System without batteries")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel('Throughput [Mb/s]')
        
        ## Only Macro
        fig_throughput_only_macro, ax = plt.subplots()
        self.list_figures.append((fig_throughput_only_macro, 'output-throughput-only-macro'))
        ax.plot(sim_times, self.output_throughput_only_macro/10e6, label="Macro Cells")
        ax.legend()
        ax.set_title("Throughput Downlink. System with only Macro Cells")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel('Throughput [Mb/s]')
        
        # Get the context_class method
        super().plot_output(sim_times=sim_times, show_plots=show_plots)
