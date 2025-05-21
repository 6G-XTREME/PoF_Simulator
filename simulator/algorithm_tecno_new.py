__author__ = "Enrique Fernandez Sanchez (efernandez@e-lighthouse.com)"
__credits__ = ["Enrique Fernandez Sanchez"]
__version__ = "1.2"
__maintainer__ = "Enrique Fernandez Sanchez"
__email__ = "efernandez@e-lighthouse.com"
__status__ = "Validated"

import numpy as np
import pandas as pd
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from tqdm import tqdm
import matplotlib.pyplot as plt
import time, random, os

from simulator.launch import logger
from simulator.context_config import Contex_Config
from simulator.solar_harvesting import SolarPanel, Weather
import simulator.map_utils, simulator.user_association_utils, simulator.radio_utils, simulator.energy_utils
from run_simulator_technoeconomics import CONFIG_PARAMETERS

from simulator.user_association_utils import search_closest_macro
from simulator.map_utils import search_closest_bs_optimized
from model.TrafficModel import estimate_traffic_from_seconds
from matplotlib.lines import Line2D
from model.TechnoEconomics import FileWithKPIs



class PoF_simulation_ELighthouse_TecnoAnalysis(Contex_Config):
    SMA_WINDOW: int
    map_scale: int                              # 1 km == 100 points (1:100) 
    
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
    
    # Pre-computed valley-spoke factors for each user and time step
    valley_spoke_factors: np.array
    
    # Percentage Vars
    per_served_femto: float
    per_in_area: float
    per_time_served: float
    
    att_db_per_km: float                    # Fiber attenuation in dB/Km
    
    # Battery Vars
    first_batt_dead_s: float
    last_batt_dead_s: float
    remaining_batt: int
    # Solar harvesting
    use_harvesting: bool                    # Enable Solar Harvesting
    weather: str                            # Select the weather condition
    city: str                               # Selected city to simulate the harvesting
    solar_panel: SolarPanel                 # Helper Object to obtain the charging intensity
    battery_mean_harvesting: np.array
    cumulative_harvested_power: np.array
    
    # Centroid Charging
    use_centroid: bool
    selected_random_macro: int              # Index of the selected macro cell
    centroid_x: float                       # X Coordinate of the centroid
    centroid_y: float                       # Y Coordinate of the centroid
    
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
    
    # Config times
    femto_boot_time_seconds: int
    femto_shutdown_time_seconds: int
    time_to_shutdown_unused_femto: int

    rng_seed: int
    max_batteries_charging: int
    charging_battery_threshold: float
    
    
    # Technoeconomics variables
    served_users_sim: np.array
    blocked_users_traffic_bps: np.array
    
    numberOfPofPools: int
    
    
    
    # ------------------------------------------------------------------------------------------------------------ #
    # -- INITIALIZATION ------------------------------------------------------------------------------------------ #
    #                                                                                                              #
    #                                                                                                              #
    #                                                                                                              #
    # ------------------------------------------------------------------------------------------------------------ #
    def __init__(self, sim_times, basestation_data: dict, user_data: dict, battery_data: dict, transmit_power_data: dict, elighthouse_parameters: dict, run_name: str, output_folder: str) -> None:
        # Set seed for random
        #random.seed(150)
        
        # Validation of custom parameters
        try:
            # Number of timeSlots that user should wait to re-send the position
            if elighthouse_parameters['user_report_position'] > 0 and elighthouse_parameters['user_report_position'] < 100:
                self.user_report_position = elighthouse_parameters['user_report_position']
            else:
                self.user_report_position = 1   # For each timeStep, the user report his position
            
            # Number of timeSlots to startup a femtocell 
            if elighthouse_parameters['startup_max_tokens'] > 0 and elighthouse_parameters['startup_max_tokens'] < 100:
                self.startup_max_tokens = elighthouse_parameters['startup_max_tokens']
            else:
                self.startup_max_tokens = 1
               
            # Number of timeSlots to Poweroff a non used Cell
            if elighthouse_parameters['poweroff_unused_cell'] > 0 and elighthouse_parameters['poweroff_unused_cell'] < 100:
                self.poweroff_max_tokens = elighthouse_parameters['poweroff_unused_cell']
            else:
                self.poweroff_max_tokens = 1

            # Solar Harvesting + PoF
            if elighthouse_parameters.get('use_harvesting', False):
                self.use_harvesting = True
                # Reference Solar Panel:
                # SeedStudio Panel: https://www.seeedstudio.com/Solar-Panel-PV-12W-with-mounting-bracket-p-5003.html
                # https://www.mouser.es/new/seeed-studio/seeed-studio-pv-12w-solar-panel/
                self.solar_panel = SolarPanel(power_rating=12, voltage_charging=14, efficiency=0.2, area=(0.35 * 0.25))
                
                valid_enum_member_names = {member.name for member in Weather}
                if elighthouse_parameters.get('weather', "") in valid_enum_member_names:
                    self.weather = elighthouse_parameters.get('weather')
                else:
                    self.weather = "SUNNY"
                
                if elighthouse_parameters.get('city', "") in self.solar_panel.irradiance_city:
                    self.city = elighthouse_parameters.get('city')
                else:
                    self.city = "Cartagena"
                logger.info("Using solar harvesting, located at city: " + self.city)
            else:
                self.use_harvesting = False
                
            # Map Scale
            self.map_scale = elighthouse_parameters.get('MapScale', 100)
               
            # Fiber dB attenuation per Km 
            if elighthouse_parameters.get('fiberAttdBperKm') > 0.0 and elighthouse_parameters.get('fiberAttdBperKm') <= 0.4:
                self.att_db_per_km = elighthouse_parameters.get('fiberAttdBperKm', 0.2)
            else:
                self.att_db_per_km = 0.2
                
            # Centroid Based Charging
            if elighthouse_parameters.get('extraPoFCharger', False):
                self.use_centroid = True
                if 'centroid_x' in elighthouse_parameters:
                    self.centroid_x = elighthouse_parameters['centroid_x']
                else:
                    logger.error("Centroid mode is enabled but its unable to found the X coordinate")
                    raise Exception
                if 'centroid_x' in elighthouse_parameters:
                    self.centroid_y = elighthouse_parameters['centroid_y']    
                else:
                    logger.error("Centroid mode is enabled but its unable to found the Y coordinate")
                    raise Exception
                
                if 'selected_random_macro' in elighthouse_parameters:
                    self.selected_random_macro = elighthouse_parameters['selected_random_macro']
                else:
                    self.selected_random_macro = -1
                    
                logger.info(f"Using centroids: X={self.centroid_x}, Y={self.centroid_y}")
            else:
                self.use_centroid = False
                
            # Config times
            if elighthouse_parameters.get('config_times', None) is not None:
                self.femto_boot_time_seconds = elighthouse_parameters['config_times']['femto_boot_time_seconds']
                self.femto_shutdown_time_seconds = elighthouse_parameters['config_times']['femto_shutdown_time_seconds']
                self.time_to_shutdown_unused_femto = elighthouse_parameters['config_times']['time_to_shutdown_unused_femto']
                
                
                
                
        except Exception as ex:
            # On error, load default custom parameters
            logger.error(f"Unable to validate custom parameters, insted using defaults. Error: {ex}")
            self.user_report_position = 1
            self.startup_max_tokens = 1
            self.poweroff_max_tokens = 1
            self.use_harvesting = False
            self.weather = "SUNNY"
            self.city = "Cartagena"
            self.map_scale = 100
            self.att_db_per_km = 0.2
            self.use_centroid = False
            self.selected_random_macro = -1
        
        super().__init__(sim_times=sim_times, basestation_data=basestation_data, user_data=user_data, battery_data=battery_data, transmit_power_data=transmit_power_data)
        
        # Initialize valley_spoke_factors as None - will be populated in start_simulation
        self.valley_spoke_factors = None
        self.run_name = run_name
        self.rng_seed = CONFIG_PARAMETERS.get('rng_seed', 1234567890)
        self.rng = random.Random(self.rng_seed)
        self.charging_battery_threshold = CONFIG_PARAMETERS.get('charging_battery_threshold', 0.95)
        
        # Obtain the number of batteries that can be charging simultaneously
        simultaneous_charging_batteries = CONFIG_PARAMETERS.get('simultaneous_charging_batteries', "40%")
        if simultaneous_charging_batteries == "ALL":
            self.max_batteries_charging = self.NFemtoCells
        elif simultaneous_charging_batteries.endswith("%"):
            self.max_batteries_charging = int(self.NFemtoCells * float(simultaneous_charging_batteries.replace("%", "")) / 100)
        else:
            self.max_batteries_charging = int(simultaneous_charging_batteries)
        
        # SMA Window
        self.SMA_WINDOW = CONFIG_PARAMETERS.get('SMA_WINDOW', 10)
        
        
        # Number of POF Pools
        self.numberOfPofPools = elighthouse_parameters.get('numberOfPofPools', 1)
        
        super().create_folders(run_name, output_folder)


    
    
    # ------------------------------------------------------------------------------------------------------------ #
    # -- SIMULATION ---------------------------------------------------------------------------------------------- #
    #                                                                                                              #
    #                                                                                                              #
    #                                                                                                              #
    # ------------------------------------------------------------------------------------------------------------ #
    def start_simulation(self, sim_times, timeStep, text_plot, progressbar_widget, canvas_widget, show_plots: bool = True, speed_plot: float = 0.05):
        # Store timeStep as class attribute
        self.timeStep = timeStep
        
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
        
        # Solar Harvesting
        self.battery_mean_harvesting = np.zeros(len(sim_times))
        self.cumulative_harvested_power = np.zeros(len(self.battery_vector[0]))
        
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
        
        # Pre-compute valley-spoke factors for each user and time step
        self.valley_spoke_factors = np.zeros((len(sim_times), len(self.NUsers)))
        for timeIndex in range(len(sim_times)):
            for userIndex in range(len(self.NUsers)):
                self.valley_spoke_factors[timeIndex, userIndex] = estimate_traffic_from_seconds(timeIndex * timeStep, ruido_max=0.05)

                
                
        self.served_users_sim = np.zeros((len(sim_times), len(self.NUsers)), dtype=int) # 0: served by femto, 1: served by macro
        self.blocked_users_traffic_bps = np.zeros((len(sim_times), len(self.NUsers)), dtype=float)
        
        logger.info("Starting simulation...")
        start = time.time()
        # Progress Bar
        with tqdm(total=100, desc='Simulating...') as f:
            for timeIndex in range(len(sim_times)):
                if f.n > f.total:
                    break

                # Update progress bar and message
                stage = timeIndex * 100 / len(sim_times)+1
                f.update(100 / len(sim_times))
                f.set_description("%.2f %% completed..." % (stage))

                if canvas_widget is None:   # Only show time in the plot when is outside de UI
                    t = sim_times[timeIndex]
                    text_plot.set_text('Time (sec) = {:.2f}'.format(t))

                self.algorithm_step(timeIndex=timeIndex, timeStep=timeStep)
                self.compute_statistics_for_plots(timeIndex=timeIndex)                          # Prepare derivate data for plots
                self.update_battery_state(timeIndex=timeIndex, timeStep=timeStep)               # Update battery state for next timeStep
                
                if progressbar_widget is not None: 
                    progressbar_widget.setValue(int(stage))
                
                if show_plots:
                    if canvas_widget is None:
                        plt.draw()
                        plt.pause(speed_plot)
                if canvas_widget is not None: 
                    canvas_widget.draw()
        
                
        # Finished
        logger.info("Simulation complete!")
        logger.info(f"Elapsed time: {np.round(time.time() - start, decimals=4)} seconds.")
        return

 
 
 
    # ------------------------------------------------------------------------------------------------------------ #
    # -- ALGORITHM ----------------------------------------------------------------------------------------------- #
    #                                                                                                              #
    #                                                                                                              #
    #                                                                                                              #
    # ------------------------------------------------------------------------------------------------------------ #

    # Helper: Set user association line
    def _set_user_association_line(self, userIndex, X, Y, color, linestyle, linewidth):
        self.user_association_line[userIndex].set_data(X, Y)
        self.user_association_line[userIndex].set_color(color)
        self.user_association_line[userIndex].set_linestyle(linestyle)
        self.user_association_line[userIndex].set_linewidth(linewidth)

    # Helper: Find closest macro
    def _find_closest_macro(self, userIndex, timeIndex):
        return search_closest_macro(
            [self.user_list[userIndex]["v_x"][timeIndex], self.user_list[userIndex]["v_y"][timeIndex]],
            self.BaseStations[0:self.NMacroCells, 0:2]
        )

    # Helper: Associate user to macro
    def _associate_to_macro(self, userIndex, timeIndex, macro_index, color='orange', linestyle='--', linewidth=0.5):
        X = [self.user_list[userIndex]["v_x"][timeIndex], self.BaseStations[macro_index, 0]]
        Y = [self.user_list[userIndex]["v_y"][timeIndex], self.BaseStations[macro_index, 1]]
        self._set_user_association_line(userIndex, X, Y, color, linestyle, linewidth)
        self.association_vector[0, userIndex] = macro_index
        self.is_in_femto[userIndex][timeIndex] = 2
        self.active_Cells[timeIndex][macro_index] = 1
        self.baseStation_users[timeIndex][macro_index] += 1

    # Helper: Associate user to femto
    def _associate_to_femto(self, userIndex, timeIndex, femto_index, color, linestyle, linewidth, discharging=False, overflow_macro=None):
        X = [self.user_list[userIndex]["v_x"][timeIndex], self.BaseStations[femto_index, 0]]
        Y = [self.user_list[userIndex]["v_y"][timeIndex], self.BaseStations[femto_index, 1]]
        self._set_user_association_line(userIndex, X, Y, color, linestyle, linewidth)
        self.association_vector[0, userIndex] = femto_index
        self.is_in_femto[userIndex][timeIndex] = 1
        self.baseStation_users[timeIndex][femto_index] += 1
        if discharging and overflow_macro is not None:
            self.association_vector_overflow_alternative[0, userIndex] = overflow_macro
            self.overflown_from[timeIndex][femto_index] += 1
        else:
            self.association_vector_overflow_alternative[0, userIndex] = 0

    def algorithm_step(self, timeIndex, timeStep):
        """ Algorithm Logic to execute in each timeStep of the simulation
        Args:
            timeIndex (int): Current simulation time step
            timeStep (float): Time step duration in seconds
        """
        if timeIndex == 0:
            self.battery_state[timeIndex] = np.zeros(self.NMacroCells+self.NFemtoCells)
        if self.started_up_femto is None:
            self.started_up_femto = []
        try:
            self.battery_state[timeIndex+1] = np.zeros(self.NMacroCells+self.NFemtoCells)
            self.baseStation_users[timeIndex+1] = np.zeros(self.NMacroCells+self.NFemtoCells)
            self.active_Cells[timeIndex+1] = np.zeros(self.NMacroCells+self.NFemtoCells)
            self.overflown_from[timeIndex+1] = np.zeros(self.NMacroCells+self.NFemtoCells)
            self.temporal_association_vector = np.zeros(self.NMacroCells, dtype=int)
        except:
            pass
        self.baseStation_users[timeIndex] = np.zeros(self.NMacroCells+self.NFemtoCells)
        self.active_Cells[timeIndex] = np.zeros(self.NMacroCells+self.NFemtoCells)
        self.overflown_from[timeIndex] = np.zeros(self.NMacroCells+self.NFemtoCells)

        # Calculate boot/shutdown steps based on time granularity
        boot_steps = max(1, int(self.femto_boot_time_seconds / timeStep))
        shutdown_steps = max(1, int(self.femto_shutdown_time_seconds / timeStep))
        unused_shutdown_steps = max(1, int(self.time_to_shutdown_unused_femto / timeStep))

        for userIndex in range(0, len(self.NUsers)):
            self.user_pos_plot[userIndex][0].set_data([
                self.user_list[userIndex]["v_x"][timeIndex],
                self.user_list[userIndex]["v_y"][timeIndex]
            ])
            user_position = [self.user_list[userIndex]["v_x"][timeIndex], self.user_list[userIndex]["v_y"][timeIndex]]
            closest_bs_dl = search_closest_bs_optimized(user_position, self.Regions, self.BaseStations, self.NMacroCells)
            self.user_closest_bs[timeIndex][userIndex] = closest_bs_dl

            def is_femtocell(bs_index):
                return bs_index >= self.NMacroCells

            if is_femtocell(closest_bs_dl):
                if self.baseStation_users[timeIndex][closest_bs_dl] == 0:
                    active_femto = np.sum(self.active_Cells[timeIndex][self.NMacroCells:])
                    battery_femto = np.count_nonzero(self.battery_state[timeIndex][self.NMacroCells:] == 2.0)

                    current_watts = (active_femto * self.small_cell_consumption_ON) + ((self.NFemtoCells - (active_femto + battery_femto)) * self.small_cell_consumption_SLEEP)
                    current_battery = self.battery_vector[0, closest_bs_dl]
                    estimated_consumption = (timeStep/3600) * self.small_cell_current_draw
                    freedom_degree = 0.4
                    enough_battery = current_battery > estimated_consumption * (1 - freedom_degree)
                    enough_battery = True

                    if current_watts >= (self.max_energy_consumption_active - self.small_cell_consumption_ON + self.small_cell_consumption_SLEEP):
                        if enough_battery:
                            # If timeStep is greater than boot time, activate immediately
                            if timeStep >= self.femto_boot_time_seconds:
                                if closest_bs_dl not in self.started_up_femto:
                                    self.started_up_femto.append(closest_bs_dl)
                                macro = self._find_closest_macro(userIndex, timeIndex)
                                self._associate_to_femto(userIndex, timeIndex, closest_bs_dl, 'green', '--', 3, discharging=True, overflow_macro=macro)
                                self.active_Cells[timeIndex][closest_bs_dl] = 0
                                self.battery_state[timeIndex][closest_bs_dl] = 2.0
                                self.battery_vector[0, closest_bs_dl] = max(0, self.battery_vector[0, closest_bs_dl] - estimated_consumption)
                                continue
                            else:
                                # Handle boot sequence based on time granularity
                                if (self.starting_up_femto[closest_bs_dl] > 0 and self.starting_up_femto[closest_bs_dl] <= boot_steps):
                                    self.active_Cells[timeIndex][closest_bs_dl] = 0
                                    self.battery_state[timeIndex][closest_bs_dl] = 2.0
                                elif closest_bs_dl in self.started_up_femto:
                                    macro = self._find_closest_macro(userIndex, timeIndex)
                                    self._associate_to_femto(userIndex, timeIndex, closest_bs_dl, 'green', '--', 3, discharging=True, overflow_macro=macro)
                                    self.active_Cells[timeIndex][closest_bs_dl] = 0
                                    self.battery_state[timeIndex][closest_bs_dl] = 2.0
                                    self.battery_vector[0, closest_bs_dl] = max(0, self.battery_vector[0, closest_bs_dl] - estimated_consumption)
                                    continue
                                else:
                                    self.starting_up_femto[closest_bs_dl] = boot_steps
                                    self.active_Cells[timeIndex][closest_bs_dl] = 0
                                    self.battery_state[timeIndex][closest_bs_dl] = 2.0
                            macro = self._find_closest_macro(userIndex, timeIndex)
                            self._associate_to_macro(userIndex, timeIndex, macro, color='orange', linestyle='--', linewidth=0.5)
                        else:
                            macro = self._find_closest_macro(userIndex, timeIndex)
                            self._associate_to_macro(userIndex, timeIndex, macro, color='red', linestyle='--', linewidth=2)
                    else:
                        # If timeStep is greater than boot time, activate immediately
                        if timeStep >= self.femto_boot_time_seconds:
                            if closest_bs_dl not in self.started_up_femto:
                                self.started_up_femto.append(closest_bs_dl)
                            self._associate_to_femto(userIndex, timeIndex, closest_bs_dl, self.colorsBS[closest_bs_dl], '-', 0.5)
                            self.active_Cells[timeIndex][closest_bs_dl] = 1
                            self.battery_state[timeIndex][closest_bs_dl] = 0.0
                            continue
                        else:
                            # Handle boot sequence based on time granularity
                            if (self.starting_up_femto[closest_bs_dl] > 0 and self.starting_up_femto[closest_bs_dl] <= boot_steps):
                                self.active_Cells[timeIndex][closest_bs_dl] = 1
                                self.battery_state[timeIndex][closest_bs_dl] = 0.0
                            elif closest_bs_dl in self.started_up_femto:
                                self._associate_to_femto(userIndex, timeIndex, closest_bs_dl, self.colorsBS[closest_bs_dl], '-', 0.5)
                                self.active_Cells[timeIndex][closest_bs_dl] = 1
                                self.battery_state[timeIndex][closest_bs_dl] = 0.0
                                continue
                            else:
                                self.starting_up_femto[closest_bs_dl] = boot_steps
                                self.active_Cells[timeIndex][closest_bs_dl] = 1
                                self.battery_state[timeIndex][closest_bs_dl] = 0.0
                        macro = self._find_closest_macro(userIndex, timeIndex)
                        self._associate_to_macro(userIndex, timeIndex, macro, color='orange', linestyle='--', linewidth=0.5)
                else:
                    if self.starting_up_femto[closest_bs_dl] == 0:
                        discharging = self.battery_state[timeIndex][closest_bs_dl] == 2.0
                        if discharging:
                            macro = self._find_closest_macro(userIndex, timeIndex)
                            self._associate_to_femto(userIndex, timeIndex, closest_bs_dl, 'green', '--', 3, discharging=True, overflow_macro=macro)
                        else:
                            self._associate_to_femto(userIndex, timeIndex, closest_bs_dl, self.colorsBS[closest_bs_dl], '-', 0.5)
                    else:
                        macro = self._find_closest_macro(userIndex, timeIndex)
                        self._associate_to_macro(userIndex, timeIndex, macro, color='orange', linestyle='--', linewidth=0.5)
            else:
                X = [self.user_list[userIndex]["v_x"][timeIndex], self.BaseStations[closest_bs_dl, 0]]
                Y = [self.user_list[userIndex]["v_y"][timeIndex], self.BaseStations[closest_bs_dl, 1]]
                self._set_user_association_line(userIndex, X, Y, self.colorsBS[closest_bs_dl], '-', 0.5)
                self.association_vector[0, userIndex] = closest_bs_dl
                self.association_vector_overflow_alternative[0, userIndex] = 0
                self.active_Cells[timeIndex][closest_bs_dl] = 0
                self.baseStation_users[timeIndex][closest_bs_dl] += 1

        # End user allocation in timeIndex instance
        # Check for unused femtocells based on time granularity
        if timeIndex % unused_shutdown_steps == 0:
            for femto in range(0, len(self.started_up_femto)):
                try:
                    if self.baseStation_users[timeIndex][self.started_up_femto[femto]] == 0:
                        # If timeStep is greater than shutdown time, deactivate immediately
                        if timeStep >= self.femto_shutdown_time_seconds:
                            self.started_up_femto.remove(self.started_up_femto[femto])
                        else:
                            # Start shutdown sequence
                            self.starting_up_femto[self.started_up_femto[femto]] = -shutdown_steps
                except:
                    pass

        # Update femtocell states
        for femto in range(0, len(self.starting_up_femto)):
            if self.starting_up_femto[femto] > 0:
                if self.starting_up_femto[femto] == 1:
                    try:
                        self.started_up_femto.append(femto)
                    except:
                        self.started_up_femto = []
                        self.started_up_femto.append(femto)
                self.starting_up_femto[femto] = self.starting_up_femto[femto] - 1
            elif self.starting_up_femto[femto] < 0:
                if self.starting_up_femto[femto] == -1:
                    try:
                        self.started_up_femto.remove(femto)
                    except:
                        pass
                self.starting_up_femto[femto] = self.starting_up_femto[femto] + 1

        # Calculate traffic for all users
        for userIndex in range(0, len(self.NUsers)):
            self.X_user[timeIndex][userIndex][0] = self.calculate_traffic(userIndex=userIndex, timeIndex=timeIndex, timeStep=timeStep)
            self.X_user[timeIndex][userIndex][1] = self.calculate_traffic_no_battery(userIndex=userIndex, timeIndex=timeIndex, timeStep=timeStep)
            self.X_user[timeIndex][userIndex][2] = self.calculate_traffic_only_macro(userIndex=userIndex, timeIndex=timeIndex, timeStep=timeStep)

        return





    # ------------------------------------------------------------------------------------------------------------ #
    # -- DEBUG PLOTS --------------------------------------------------------------------------------------------- #
    #                                                                                                              #
    #                                                                                                              #
    #                                                                                                              #
    # ------------------------------------------------------------------------------------------------------------ #
    def save_battery_capacity_debug_plot(self, timeIndex):
        """Creates and saves a debug plot showing the current battery capacity state
        
        Args:
            timeIndex (int): Current simulation time step
        """
        # Create debug plot with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[2, 1])
        
        # Top subplot - Battery capacity
        ax1.axhline(y=3.3, color='r', label="Max. capacity")
        
        # Get battery states for coloring
        battery_states = self.battery_state[timeIndex]
        
        # Plot each femtocell's battery
        for bar in range(self.NMacroCells, len(self.battery_vector[0])):
            # Determine color based on battery state
            if battery_states[bar] == 0:  # Nothing
                color = 'gray'
            elif battery_states[bar] == 1:  # Charging
                color = 'green'
            elif battery_states[bar] == 2:  # Discharging
                color = 'red'
            else:  # Discharging & Charging
                color = 'orange'
                
            ax1.bar(int(bar), self.battery_vector[0][bar]*1000, color=color)
        
        # Add legend for battery states
        ax1.bar(0, 0, color='gray', label='Inactive')
        ax1.bar(0, 0, color='green', label='Charging')
        ax1.bar(0, 0, color='red', label='Discharging')
        ax1.bar(0, 0, color='orange', label='Charging & Discharging')
        ax1.axhline(y=3.3, color='r', label="Max. capacity")
        
        ax1.legend()
        ax1.set_title(f"Battery Capacity at Time Step {timeIndex}")
        ax1.set_xlabel("Femto cell number")
        ax1.set_ylabel("Capacity [mAh]")

        # Bottom subplot - Femtocell status
        femto_indices = np.arange(len(self.battery_state[timeIndex][self.NMacroCells:]))
        status_colors = []
        status_labels = []
        
        for i in range(len(femto_indices)):
            femto_idx = i + self.NMacroCells
            # Check various femtocell states
            if femto_idx in self.started_up_femto:
                color = 'green'
                label = 'Active'
            elif self.starting_up_femto[femto_idx] > 0:
                color = 'yellow'
                label = 'Booting'
            elif self.starting_up_femto[femto_idx] < 0:
                color = 'orange' 
                label = 'Shutting Down'
            else:
                color = 'black'
                label = 'Inactive'

            status_colors.append(color)
            status_labels.append(label)
            
        # Create bar plot
        ax2.bar(femto_indices, [1]*len(femto_indices), color=status_colors)
        
        # Add legend with unique statuses
        unique_colors = list(set(zip(status_colors, status_labels)))
        for color, label in unique_colors:
            ax2.bar(0, 0, color=color, label=label)
            
        ax2.set_ylim(-0.1, 1.1)
        ax2.set_yticks([0, 1])
        ax2.set_yticklabels(['Off', 'On'])
        ax2.set_xlabel("Femto cell number")
        ax2.set_title("Femtocell Status")
        ax2.legend()
        ax2.grid(True)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(os.path.join(self.debug_battery_folder, f'battery_capacity_step_{timeIndex:04d}.png'), dpi=200, bbox_inches='tight')
        plt.close(fig)
    
    
    
    
    
    
    
    
    
    
    
    
    # ------------------------------------------------------------------------------------------------------------ #
    # -- STATISTICS ---------------------------------------------------------------------------------------------- #
    #                                                                                                              #
    #                                                                                                              #
    #                                                                                                              #
    # ------------------------------------------------------------------------------------------------------------ #
    def compute_statistics_for_plots(self, timeIndex):
        """ Compute Statistics in order to Plot the Algorithm Output

        Args:
            timeIndex (_type_): timeSim in this moment
        """

        # Number of active Smallcells
        self.live_smallcell_occupancy[timeIndex] = np.count_nonzero(self.active_Cells[timeIndex][self.NMacroCells:]) + 1    # +1 in order to match MATLAB

        # Cells that overflow
        self.live_smallcell_overflow[timeIndex] = np.count_nonzero(self.overflown_from[timeIndex][self.NMacroCells:]) + 1   # +1 in order to match MATLAB

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


        # Update the served users and blocked users
        for userIndex in range(0, len(self.NUsers)):
            if self.association_vector[0][userIndex] < self.NMacroCells:
                self.served_users_sim[timeIndex][userIndex] = 1
                self.blocked_users_traffic_bps[timeIndex][userIndex] = self.X_user[timeIndex][userIndex][0]
            # else: already 0
                
                
                

        
        
        return





    # ------------------------------------------------------------------------------------------------------------ #
    # -- UPDATE BATTERY STATE ------------------------------------------------------------------------------------ #
    #                                                                                                              #
    #                                                                                                              #
    #                                                                                                              #
    # ------------------------------------------------------------------------------------------------------------ #
    def update_battery_state(self, timeIndex, timeStep):
        """ Legacy function to update battery state

        Args:
            timeIndex (_type_): Index of simulation times
            timeStep (_type_): actual simulation step
        """

        # First, add the Solar harvesting for each battery of femtocell ...
        if self.use_harvesting:
            for batt in range(self.NMacroCells, len(self.battery_vector[0])):
                if self.battery_vector[0][batt] < self.battery_capacity:
                    charging_power_amperes_timeStep = self.solar_panel.calculate_Ah_in_timeStep(
                        solar_irradiance=self.solar_panel.irradiance_city[self.city],
                        timeStep=timeStep,
                        weather_condition=Weather[self.weather]
                    )
                    self.battery_vector[0][batt] = min(self.battery_vector[0][batt] + charging_power_amperes_timeStep, self.battery_capacity)
                    self.cumulative_harvested_power[batt] += charging_power_amperes_timeStep
            self.battery_mean_harvesting[timeIndex] = np.mean(self.cumulative_harvested_power)

        # extra_budget = total-active
        # live_energy < max_energy_active:
        #   available = max_energy_active - live_energy + extra_budget [big number]
        # else:
        #  using batteries, we are in the max usage of PoF, but still something left
        #   available = extra_budget [not too big]
        
        # Decide about battery recharging
        extra_budget = self.max_energy_consumption_total - self.max_energy_consumption_active
        if self.live_smallcell_consumption[timeIndex]  < self.max_energy_consumption_active:
            # Asign available energy to charge a cell battery
            available = (self.max_energy_consumption_active - self.live_smallcell_consumption[timeIndex]) + extra_budget
        else:
            # The live consumption exceed the PoF active limit, so we are on the max usage of the Budget (max_energy_consumption_active)
            available = extra_budget            # Only available the extra energy in the PoF Budget
        
        #print(f"Total: {self.max_energy_consumption_total}, Max_active: {self.max_energy_consumption_active}")
        #print(f"Live active: {self.live_smallcell_consumption[timeIndex]}")
        #print(f"Extra budget: {extra_budget}. Available: {available}")
        
        # Select the batteries to be charged with PoF, based on the battery level and the maximum number of batteries that can be charged simultaneously
        charging_threshold = self.charging_battery_threshold * self.battery_capacity   # Limit of the battery level to be charged
        battery_levels = self.battery_vector[0][self.NMacroCells:]                      # Battery levels of the femtocells
        eligible_relative_indices = np.where(battery_levels < charging_threshold)[0]      # Indices of the batteries that are eligible to be charged

        # Create list of (index, level) tuples and shuffle to randomize ties
        battery_tuples = [(i, battery_levels[i]) for i in eligible_relative_indices]
        self.rng.shuffle(battery_tuples)  # Randomize before sorting to break ties randomly
        
        # Sort by level (ascending) and limit to max number that can charge
        sorted_by_level = [i for i, _ in sorted(battery_tuples, key=lambda x: x[1])]
        selected_indices = [self.NMacroCells + i for i in sorted_by_level[:self.max_batteries_charging]]
        self.rng.shuffle(selected_indices)  # Final shuffle of selected indices

        # Process batteries one by one, updating available energy after each charge
        remaining_available = available
        for i in selected_indices:
            if remaining_available <= 0:
                break  # No more energy available in the pool
                
            i_x = self.BaseStations[i][0]
            i_y = self.BaseStations[i][1]
            effective_available = remaining_available

            if self.use_centroid:
                d_km = simulator.map_utils.get_distance_in_kilometers(
                    [i_x, i_y], [self.centroid_x, self.centroid_y], self.map_scale
                )
                effective_available = simulator.energy_utils.get_power_with_attenuation(
                    remaining_available, self.att_db_per_km, d_km
                )
            else:
                # Nearest MacroStation Charging Mode
                closest_macro = simulator.user_association_utils.search_closest_macro(
                    [i_x, i_y], self.BaseStations[0:self.NMacroCells, 0:2]
                )
                macro_x = self.BaseStations[closest_macro][0]
                macro_y = self.BaseStations[closest_macro][1]
                d_km = simulator.map_utils.get_distance_in_kilometers(
                    [i_x, i_y], [macro_x, macro_y], self.map_scale
                )
                effective_available = simulator.energy_utils.get_power_with_attenuation(
                    remaining_available, self.att_db_per_km, d_km
                )

            # Convert available energy to battery charge (Ah)
            charging_intensity = effective_available / np.mean(self.small_cell_voltage_range)
            charge_added = (charging_intensity * timeStep) / 3600
            
            # Calculate how much energy was actually used
            energy_used = (charge_added * np.mean(self.small_cell_voltage_range) * 3600) / timeStep
            
            # Update battery and remaining available energy
            self.battery_vector[0][i] = min(self.battery_vector[0][i] + charge_added, self.battery_capacity)
            remaining_available -= energy_used

        self.battery_mean_values[timeIndex] = np.mean(self.battery_vector[0])

        # Check if battery is dead [Only femtocells]
        for batt in range(self.NMacroCells, len(self.battery_vector[0])):
            battery_capacity = round(self.battery_vector[0][batt], 2)
            if battery_capacity == 0:
                if self.timeIndex_first_battery_dead == 0:
                    # First battery dead
                    self.timeIndex_first_battery_dead = timeIndex
                    self.timeIndex_last_battery_dead = timeIndex
                    self.dead_batteries.append(batt)
                else:
                    # Already found the first battery_dead
                    if batt not in self.dead_batteries:
                        self.timeIndex_last_battery_dead = timeIndex
                        self.dead_batteries.append(batt)
    
    
    
    
    
    
    
    # ------------------------------------------------------------------------------------------------------------ #
    # -- TRAFFIC CALCULATION ------------------------------------------------------------------------------------- #
    #                                                                                                              #
    #                                                                                                              #
    #                                                                                                              #
    # ------------------------------------------------------------------------------------------------------------ #
    def compute_sinr_naturalDL(self, userIndex, timeIndex, station):
        """Calculates the SINR and converts it to linear form"""
        SINRDLink = simulator.radio_utils.compute_sinr_dl(
            [self.user_list[userIndex]["v_x"][timeIndex],
            self.user_list[userIndex]["v_y"][timeIndex]],
            self.BaseStations,
            station,
            self.alpha_loss,
            self.PMacroCells,
            self.PFemtoCells,
            self.NMacroCells,
            self.noise
        )
        return 10 ** (SINRDLink / 10)

    def calculate_throughput(self, _BW, _user_count, _naturalDL, _traffic_factor=1.0):
        """General throughput calculation (bps)"""
        if _user_count < 1:
            _user_count = 1
        max_bw = (_BW / _user_count)
        estm_bw = max_bw * np.log2(1 + _naturalDL)
        return min(estm_bw, max_bw) * _traffic_factor


    
    
    def calculate_traffic(self, userIndex, timeIndex, timeStep):
        """Throughput WITH batteries given an User and timeIndex
        
        Depends of:     association_vector
                        baseStation_users
                        
        Returns Traffic of User
        """
        associated_station = int(self.association_vector[0][userIndex])
        naturalDL = self.compute_sinr_naturalDL(userIndex, timeIndex, associated_station)
        valley_spoke_factor = self.valley_spoke_factors[timeIndex, userIndex]
    
        if associated_station < self.NMacroCells:
            BW = self.MacroCellDownlinkBW
            users = self.baseStation_users[timeIndex][associated_station]
            X = self.calculate_throughput(BW, users, naturalDL, valley_spoke_factor)
            self.X_macro_bps[timeIndex][associated_station] += X
        else:
            BW = self.FemtoCellDownlinkBW
            users = self.baseStation_users[timeIndex][associated_station]
            X = self.calculate_throughput(BW, users, naturalDL, valley_spoke_factor)
            self.X_femto_bps[timeIndex][associated_station] += X
    
        return X
    




    # ------------------------------------------------------------------------------------------------------------ #
    # -- TRAFFIC CALCULATION WITHOUT BATTERIES --------------------------------------------------------------- #
    #                                                                                                              #
    #                                                                                                              #
    #                                                                                                              #
    # ------------------------------------------------------------------------------------------------------------ #
    def calculate_traffic_no_battery(self, userIndex, timeIndex, timeStep):
        """Throughput WITHOUT batteries given an User and timeIndex
        
        Depends of:     association_vector_overflow_alternative
                        association_vector
                        baseStation_users
                        
        Returns Traffic of User"""
        
        associated_station_overflow = int(self.association_vector_overflow_alternative[0][userIndex])
        valley_spoke_factor = self.valley_spoke_factors[timeIndex, userIndex]
        
        if associated_station_overflow == 0:
            associated_station = int(self.association_vector[0][userIndex])
            naturalDL = self.compute_sinr_naturalDL(userIndex, timeIndex, associated_station)
    
            if associated_station < self.NMacroCells:
                BW = self.MacroCellDownlinkBW
                user_count = self.baseStation_users[timeIndex][associated_station] + \
                            np.sum(self.association_vector_overflow_alternative == associated_station_overflow)
                X = self.calculate_throughput(BW, user_count, naturalDL, valley_spoke_factor)
                self.X_macro_no_batt_bps[timeIndex][associated_station] += X
            else:
                BW = self.FemtoCellDownlinkBW
                user_count = self.baseStation_users[timeIndex][associated_station] + \
                            self.overflown_from[timeIndex][associated_station]
                X = self.calculate_throughput(BW, user_count, naturalDL, valley_spoke_factor)
                self.X_femto_no_batt_bps[timeIndex][associated_station] += X
    
        else:
            naturalDL = self.compute_sinr_naturalDL(userIndex, timeIndex, associated_station_overflow)
            BW = self.MacroCellDownlinkBW
            user_count = self.baseStation_users[timeIndex][associated_station_overflow] + \
                        np.sum(self.association_vector_overflow_alternative[0] == associated_station_overflow)
            X = self.calculate_throughput(BW, user_count, naturalDL, valley_spoke_factor)
            self.X_macro_overflow_bps[timeIndex][associated_station_overflow] += X
    
        return X
    


    # ------------------------------------------------------------------------------------------------------------ #
    # -- TRAFFIC CALCULATION ONLY MACRO -------------------------------------------------------------------------- #
    #                                                                                                              #
    #                                                                                                              #
    #                                                                                                              #
    # ------------------------------------------------------------------------------------------------------------ #
    def calculate_traffic_only_macro(self, userIndex, timeIndex, timeStep):
        """Throughput with ONLY Macrocells given an User and timeIndex
        
        Depends of:     nothing external
                        temporal_association_vector
                        
        Returns Traffic of User"""
        
        cl = simulator.user_association_utils.search_closest_macro(
            [self.user_list[userIndex]["v_x"][timeIndex],
            self.user_list[userIndex]["v_y"][timeIndex]],
            self.BaseStations[0:self.NMacroCells, 0:2]
        )
        self.temporal_association_vector[cl] += 1
    
        naturalDL = self.compute_sinr_naturalDL(userIndex, timeIndex, cl)
        valley_spoke_factor = self.valley_spoke_factors[timeIndex, userIndex]
        BW = self.MacroCellDownlinkBW
        user_count = self.temporal_association_vector[cl]
    
        X = self.calculate_throughput(BW, user_count, naturalDL, valley_spoke_factor)
        self.X_macro_only_bps[timeIndex][cl] += X
    
        return X
    
    













    # ------------------------------------------------------------------------------------------------------------ #
    # -- PLOT OUTPUT --------------------------------------------------------------------------------------------- #
    #                                                                                                              #
    #                                                                                                              #
    #                                                                                                              #
    # ------------------------------------------------------------------------------------------------------------ #
    def plot_output(self, sim_times, timeStep, is_gui: bool = False, show_plots: bool = True, fig_size: tuple = None, dpi: int = None):
        """ Override Show Plot Output

        Args:
            sim_times: Array of simulation times
            timeStep: Time step of the simulation
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

        # Battery dead?
        if self.timeIndex_first_battery_dead != 0:
            self.first_batt_dead_s = (self.timeIndex_first_battery_dead*timeStep)
            self.last_batt_dead_s = (self.timeIndex_last_battery_dead*timeStep)
            self.remaining_batt = np.count_nonzero(np.round(self.battery_vector[0])) - self.NMacroCells
            logger.info(f"Last Battery dead at timeIndex: {self.timeIndex_last_battery_dead} ({self.last_batt_dead_s/60} min)")
            logger.info(f"First Battery dead at timeIndex: {self.timeIndex_first_battery_dead} ({self.first_batt_dead_s/60} min)")
            logger.info(f"Remaining batteries {self.remaining_batt} of {len(self.battery_vector[0]) - self.NMacroCells}.")
        
        # Compute %'s
        # self.is_in_femto -> 1 == associated with femto, -> 2 == associated with macro, -> 0 == no on femto area
        sum_served_femto = 0    # % that a user is in femto area, and its associated
        sum_in_area = 0         # % that a user is in femto area (associated or not)
        sum_time_served = 0     # % that user is in femto and its has been associated with a femto
        for user in range(0, len(self.NUsers)):
            t_served_femto = np.count_nonzero(self.is_in_femto[user] == 1)
            sum_served_femto += (t_served_femto) / (len(sim_times))
            t_in_area = np.count_nonzero(self.is_in_femto[user] == 1) + np.count_nonzero(self.is_in_femto[user] == 2)
            sum_in_area += (t_in_area) / (len(sim_times))
            try:
                sum_time_served += (t_served_femto) / (t_in_area)
            except:
                sum_time_served += 0

        self.per_served_femto = np.round(((1/len(self.NUsers)) * sum_served_femto) * 100, 3)
        self.per_in_area = np.round(((1/len(self.NUsers)) * sum_in_area) * 100, 3)
        self.per_time_served = np.round(((1/len(self.NUsers)) * sum_time_served) * 100, 3)
        
        logger.info(f"% in area & served by femto: {self.per_served_femto} %")
        logger.info(f"% in area of femto: {self.per_in_area} %")
        logger.info(f"% of inside time, when user is in area and associated with femto : {self.per_time_served} %")
        
        # User Traffic
        fig_user_traffic, ax = plt.subplots(figsize=fig_size, dpi=dpi)
        self.list_figures.append((fig_user_traffic, "user-traffic"))    # In Order to save the figure on output folder
        
        metric = 0  # Default traffic
        for user in range(0, len(self.NUsers)):
            user_traffic = np.asarray([self.X_user[t][user][metric] for t in range(len(sim_times))])
            ax.plot(self.format_time_axis(ax, sim_times), user_traffic/1e6, label=f'User {user}')
        ax.set_title(f'Traffic for each user')
        ax.set_ylabel('Throughput [Mb/s]')
        
        # Batteries in use for each timeStep
        battery_charging = []
        for timeIndex in self.battery_state:
            count_2 = np.count_nonzero(timeIndex == 2.0)
            battery_charging.append(count_2)
            
        fig_battery_charging, ax = plt.subplots(figsize=fig_size, dpi=dpi)
        self.list_figures.append((fig_battery_charging, "discharging-cells"))    # In Order to save the figure on output folder
        ax.step(self.format_time_axis(ax, sim_times), battery_charging, label="Discharging Cells", color="blue")
        ax.set_ylim(0, max(battery_charging) + 3)
        ax.legend()
        ax.set_title("Discharging Battery Cells")
        ax.set_ylabel('Number of cells')
        
        # Battery capacity
        fig_batt_capacity, ax = plt.subplots(figsize=fig_size, dpi=dpi)
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
        if self.use_harvesting:
            fig_battery_mean_harvesting, ax = plt.subplots(figsize=fig_size, dpi=dpi)
            self.list_figures.append((fig_battery_mean_harvesting, "battery_mean_harvesting"))
            ax.plot(self.format_time_axis(ax, sim_times), self.battery_mean_values, '-', label='Hybrid PoF & Solar', color="tab:red")
            # TODO Faker
            ax.plot(self.format_time_axis(ax, sim_times), self.battery_mean_values - self.battery_mean_harvesting, '--', label='Only PoF', color="tab:blue")
            ax.axhline(y=3.3, color='tab:green',label="Max. battery capacity")
            ax.set_ylabel('Battery capacity [Ah]')
            ax.legend()
            
            fig_battery_acc_harvesting, ax = plt.subplots(figsize=fig_size, dpi=dpi)
            self.list_figures.append((fig_battery_acc_harvesting, "battery_accumulative_harvesting"))
            ax.plot(self.format_time_axis(ax, sim_times), self.battery_mean_harvesting, label='Accumulative battery harvesting')
            ax.set_ylabel('Battery capacity [Ah]')
            ax.set_title('Accumulative battery harvesting')
            ax.legend()
           
        ## Throughput
        fig_throughput, ax = plt.subplots(figsize=fig_size, dpi=dpi)
        self.list_figures.append((fig_throughput, 'output-throughput'))
        ax.plot(self.format_time_axis(ax, sim_times), self.output_throughput[0]/1e6, label="Macro Cells")
        ax.plot(self.format_time_axis(ax, sim_times), self.output_throughput[1]/1e6, label="Femto Cells")
        ax.plot(self.format_time_axis(ax, sim_times), self.live_throughput/1e6, label="Total")
        ax.legend()
        ax.set_title("Throughput Downlink. System with batteries")
        ax.set_ylabel('Throughput [Mb/s]')
        
        ## Throughput no battery
        fig_throughput_no_batt, ax = plt.subplots(figsize=fig_size, dpi=dpi)
        self.list_figures.append((fig_throughput_no_batt, 'output-throughput-no-batt'))
        ax.plot(self.format_time_axis(ax, sim_times), self.output_throughput_no_batt[0]/1e6, label="Macro Cells")
        ax.plot(self.format_time_axis(ax, sim_times), self.output_throughput_no_batt[1]/1e6, label="Femto Cells")
        ax.plot(self.format_time_axis(ax, sim_times), self.output_throughput_no_batt[2]/1e6, label="Femto Cells overflow")
        ax.plot(self.format_time_axis(ax, sim_times), self.live_throughput_NO_BATTERY/1e6, label="Total")
        ax.legend()
        ax.set_title("Throughput Downlink. System without batteries")
        ax.set_ylabel('Throughput [Mb/s]')
        
        ## Only Macro
        fig_throughput_only_macro, ax = plt.subplots(figsize=fig_size, dpi=dpi)
        self.list_figures.append((fig_throughput_only_macro, 'output-throughput-only-macro'))
        ax.plot(self.format_time_axis(ax, sim_times), self.output_throughput_only_macro/1e6, label="Macro Cells")
        ax.legend()
        ax.set_title("Throughput Downlink. System with only Macro Cells")
        ax.set_ylabel('Throughput [Mb/s]')









        
        # Throughput for tecnoeconomics
        
        # Throughput Gbps
        SMA_WINDOW = self.SMA_WINDOW
        timeIndex = len(sim_times)
        fig_throughput_smooth, ax = plt.subplots(figsize=fig_size, dpi=dpi)
        self.list_figures.append((fig_throughput_smooth, "throughput_smooth_tecno"))
        X = self.format_time_axis(ax, sim_times)
        Y = np.convolve(self.live_throughput/1e9, np.ones((SMA_WINDOW,))/SMA_WINDOW, mode='valid')
        ax.plot(X[:len(Y)], Y, label='Using PoF & batteries')
        # ax.legend()
        ax.set_title(f'PoF Throughput. Delta Time: {self.timeStep/3600:.1f} h')
        ax.set_ylabel('Throughput [Gbps]')

        
        # Power kWh
        fig_power_smooth, ax = plt.subplots(figsize=fig_size, dpi=dpi)
        self.list_figures.append((fig_power_smooth, "power_smooth_tecno"))
        X = self.format_time_axis(ax, sim_times)
        transformed_y = self.live_smallcell_consumption * self.timeStep / 3600 * 1e-3
        Y = np.convolve(transformed_y, np.ones((SMA_WINDOW,))/SMA_WINDOW, mode='valid')
        ax.step(X[:len(Y)], Y, label='Using PoF & batteries')
        # ax.legend()
        ax.set_ylim(0, max(Y)*1.1)
        ax.set_title(f'PoF Power consumption. Delta Time: {self.timeStep/3600:.1f} h')
        ax.set_ylabel('Power [kWh]')
        
        
        
        
        
        
        
        
        
        
        
        
        

        # Plot associations of users to cells - Femto and Macro regions
        last_user_to_bs_assoc = self.association_vector[0, :]
        users_pos = np.array([[self.user_list[user]["v_x"][-1], self.user_list[user]["v_y"][-1]] 
                             for user in range(len(self.NUsers))])

        # Separate users and lines by cell type
        p2p_lines_fem, p2p_lines_mac = [], []
        users_fem, users_mac = [], []

        # For accurate femtocell state, build a dict: femto_idx -> run_mode (0=off, 1=laser, 2=battery)
        final_timeIndex = -1  # último paso de tiempo
        femto_run_modes = {}
        for i in range(self.NMacroCells, len(self.BaseStations)):
            is_active = self.active_Cells[final_timeIndex][i] == 1 # 0 = off, 1 = active
            battery_discharging = self.battery_state[final_timeIndex][i] == 2.0 or self.battery_state[final_timeIndex][i] == 3.0 # 2 = discharging, 3 = charging and discharging, 0-1 = other
            if is_active:
                run_mode = 1  # laser
            elif battery_discharging:
                run_mode = 2  # battery
            else:
                run_mode = 0  # off
            femto_run_modes[i] = run_mode

        # Now, when separating users, check if the femtocell is actually active
        for user, bs in enumerate(last_user_to_bs_assoc):
            user_pos = (self.user_list[user]["v_x"][-1], self.user_list[user]["v_y"][-1])
            bs_pos = self.BaseStations[int(bs)][:2]
            line = ([user_pos[0], bs_pos[0]], [user_pos[1], bs_pos[1]])

            if bs >= self.NMacroCells:
                # Only associate to femto if it is active (run_mode 1 or 2)
                run_mode = femto_run_modes.get(int(bs), 0)
                if run_mode > 0:
                    p2p_lines_fem.append(line)
                    users_fem.append(user_pos)
                else:
                    # If not active, treat as macro for plotting (should not happen, but for safety)
                    p2p_lines_mac.append(line)
                    users_mac.append(user_pos)
            else:
                p2p_lines_mac.append(line)
                users_mac.append(user_pos)

        # Helper functions for region coloring
        def region_config_macro(index):
            return {
                "alpha": 0.3, 
                "edgecolor": 'black', 
                "linewidth": 0.5, 
                "linestyle": '-',
                "color": 'orange'
            }

        def region_config_femto(index, run_mode):
            # run_mode: 0=off, 1=laser, 2=battery
            color = 'gray'
            if run_mode == 1:
                color = 'orange'
            elif run_mode == 2:
                color = 'green'
            return {
                "alpha": 0.3, 
                "edgecolor": 'black', 
                "linewidth": 0.5, 
                "linestyle": '-',
                "color": color
            }

        def paint_regions_macro(_regions, _ax, cell_index):
            for region in reversed(_regions):
                if isinstance(region, (Polygon, MultiPolygon, GeometryCollection)):
                    if isinstance(region, Polygon):
                        polygons = [region]
                    elif isinstance(region, MultiPolygon):
                        polygons = region.geoms
                    else:  # GeometryCollection
                        polygons = [g for g in region.geoms if isinstance(g, Polygon)]
                    
                    for poly in polygons:
                        x, y = poly.exterior.coords.xy
                        _ax.fill(x, y, **region_config_macro(cell_index))

        def paint_regions_femto(_regions, _ax, cell_index, run_mode):
            for region in reversed(_regions):
                if isinstance(region, (Polygon, MultiPolygon, GeometryCollection)):
                    if isinstance(region, Polygon):
                        polygons = [region]
                    elif isinstance(region, MultiPolygon):
                        polygons = region.geoms
                    else:  # GeometryCollection
                        polygons = [g for g in region.geoms if isinstance(g, Polygon)]
                    
                    for poly in polygons:
                        x, y = poly.exterior.coords.xy
                        _ax.fill(x, y, **region_config_femto(cell_index, run_mode))

        def setup_plot(is_femto=True):
            fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
            self.list_figures.append((fig, f'output-last-user-association-only-{"femto" if is_femto else "macro"}'))
            if is_femto:
                ax.set_title('Last User Association - Femto Cells')
                # Custom legend for femto
                legend_elements = [
                    Line2D([0], [0], color='orange', lw=6, label='FemtoCell (Laser)'),
                    Line2D([0], [0], color='green', lw=6, label='FemtoCell (Battery)'),
                    Line2D([0], [0], color='gray', lw=6, label='FemtoCell (Off)'),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=6, label='FemtoCell Position'),
                    Line2D([0], [0], marker='+', color='red', markersize=10, linestyle='None', label='User in FemtoCell'),
                    Line2D([0], [0], marker='+', color='blue', markersize=10, linestyle='None', label='User in MacroCell'),
                    Line2D([0], [0], color='green', lw=1, label='User Association')
                ]
                ax.legend(handles=legend_elements, loc='best')
            else:
                ax.set_title('Last User Association - Macro Cells')
                legend_elements = [
                    Line2D([0], [0], color='orange', lw=6, label='MacroCell Region'),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=6, label='MacroCell Position'),
                    Line2D([0], [0], marker='+', color='red', markersize=10, linestyle='None', label='User in MacroCell'),
                    Line2D([0], [0], marker='+', color='blue', markersize=10, linestyle='None', label='User in FemtoCell'),
                    Line2D([0], [0], color='green', lw=1, label='User Association')
                ]
                ax.legend(handles=legend_elements, loc='best')
            ax.axis('off')
            fig.tight_layout()
            return fig, ax

        # Create and populate plots
        fig_fem, ax_fem = setup_plot(is_femto=True)
        fig_mac, ax_mac = setup_plot(is_femto=False)

        # Draw femto cells with new color logic, using accurate run_mode for each cell
        def draw_cell_associations_femto(start_idx, end_idx, ax, lines, active_users, inactive_users, run_modes_dict):
            for i in range(start_idx, end_idx):
                run_mode = run_modes_dict.get(i, 0)
                paint_regions_femto([self.Regions[i]], ax, i, run_mode)
                ax.scatter(*self.BaseStations[i][:2], color='black', s=10, marker='o')
                ax.annotate(str(i), (self.BaseStations[i][0], self.BaseStations[i][1]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
            for line in lines:
                ax.plot(line[0], line[1], color='green', linewidth=0.8)
            for user in active_users:
                ax.scatter(*user, color='red', s=20, marker='+', linewidths=2)
            for user in inactive_users:
                ax.scatter(*user, color='blue', s=5, marker='+', linewidths=0.5)

        def draw_cell_associations_macro(start_idx, end_idx, ax, lines, active_users, inactive_users):
            for i in range(start_idx, end_idx):
                paint_regions_macro([self.Regions[i]], ax, i)
                ax.scatter(*self.BaseStations[i][:2], color='black', s=10, marker='o')
                ax.annotate(str(i), (self.BaseStations[i][0], self.BaseStations[i][1]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
            for line in lines:
                ax.plot(line[0], line[1], color='green', linewidth=0.8)
            for user in active_users:
                ax.scatter(*user, color='red', s=20, marker='+', linewidths=2)
            for user in inactive_users:
                ax.scatter(*user, color='blue', s=5, marker='+', linewidths=0.5)

        # Draw femto cells with correct region coloring and user associations
        draw_cell_associations_femto(self.NMacroCells, len(self.BaseStations), ax_fem, 
                                     p2p_lines_fem, users_fem, users_mac, femto_run_modes)
        
        # Draw macro cells (all orange)
        draw_cell_associations_macro(0, self.NMacroCells, ax_mac,
                                     p2p_lines_mac, users_mac, users_fem)

        # Get the context_class method
        super().plot_output(sim_times=sim_times, show_plots=show_plots, is_gui=is_gui, fig_size=fig_size)











    # ------------------------------------------------------------------------------------------------------------ #
    # -- SAVE RUN ------------------------------------------------------------------------------------------------ #
    #                                                                                                              #
    #                                                                                                              #
    #                                                                                                              #
    # ------------------------------------------------------------------------------------------------------------ #
    def save_run(self, fig_map, sim_times, run_name, output_folder, dpi: int = 200):
        # Legacy algorithm save
        super().save_run(fig_map, sim_times, run_name, output_folder, dpi)

        csv = os.path.join(self.data_folder, f'{run_name}-output.csv')
        json = os.path.join(self.data_folder, f'{run_name}-output.json')
        kpis_json = os.path.join(self.data_folder, f'{run_name}-kpis.json')

        # Get techno-economics metrics
        tecno_metrics = self.get_techno_economics_metrics()

        # Create FileWithKPIs object
        kpis = FileWithKPIs(
            total_throughput_gbps=tecno_metrics['total_throughput_gbps'],
            daily_avg_throughput_gbps=tecno_metrics['daily_avg_throughput_gbps'],
            total_power_consumption_kWh=tecno_metrics['total_power_consumption_kWh'],
            daily_avg_power_consumption_kWh=tecno_metrics['daily_avg_power_consumption_kWh'],
            yearly_power_estimate_kWh=tecno_metrics['yearly_power_estimate_kWh'],
            availability_percentage=tecno_metrics['availability_percentage'],
            blocked_traffic_gbps=tecno_metrics['blocked_traffic_gbps'],
            throughput_time_series_gbps=tecno_metrics['throughput_time_series_gbps'],
            power_time_series_kWh=tecno_metrics['power_time_series_kWh']
        )

        # Save KPIs to JSON using FileWithKPIs format
        kpis.to_file(kpis_json)

        # Read CSV and add the new parameters to save!
        df_update = pd.read_csv(csv)
        df_update = df_update.assign(per_served_femto=self.per_served_femto)
        df_update = df_update.assign(per_in_area=self.per_in_area)
        df_update = df_update.assign(per_time_served=self.per_time_served)
        
        # Add techno-economics metrics
        df_update = df_update.assign(total_throughput_gbps=tecno_metrics['total_throughput_gbps'])
        df_update = df_update.assign(daily_avg_throughput_gbps=tecno_metrics['daily_avg_throughput_gbps'])
        df_update = df_update.assign(total_power_consumption_kWh=tecno_metrics['total_power_consumption_kWh'])
        df_update = df_update.assign(daily_avg_power_consumption_kWh=tecno_metrics['daily_avg_power_consumption_kWh'])
        df_update = df_update.assign(yearly_power_estimate_kWh=tecno_metrics['yearly_power_estimate_kWh'])
        df_update = df_update.assign(availability_percentage=tecno_metrics['availability_percentage'])
        df_update = df_update.assign(blocked_traffic_gbps=tecno_metrics['blocked_traffic_gbps'])
        
        try:
            df_update = df_update.assign(first_batt_dead=self.first_batt_dead_s)
            df_update = df_update.assign(last_batt_dead=self.last_batt_dead_s)
            df_update = df_update.assign(remaining_batt=self.remaining_batt)

            df_update = df_update.rename(columns={'first_batt_dead': 'first_batt_dead[s]',
                                                  'last_batt_dead': 'last_batt_dead[s]'})
        except:
            pass
        
        try:
            df_update = df_update.assign(harvesting=self.use_harvesting)
            if self.use_harvesting:
                df_update = df_update.assign(weather=self.weather)
                df_update = df_update.assign(city=self.city)
            df_update = df_update.assign(centroids=self.use_centroid)
            
            if self.selected_random_macro >= 0:
                df_update = df_update.assign(selected_macro_charge=self.selected_random_macro)
        except:
            pass
        
        # Save to CSV & JSON        
        df_update.to_csv(csv, index=False)
        df_update.to_json(json, orient="index", indent=4)


    def get_techno_economics_metrics(self):
        """Get the final techno-economics metrics
        
        Returns:
            dict: Dictionary containing all techno-economics metrics
        """
        # Calculate total and average throughput

        total_throughput_gbps = sum(self.live_throughput / 1e9 )
        total_seconds = len(self.live_throughput) * self.timeStep
        num_days = total_seconds / (24 * 3600)
        daily_avg_throughput_gbps = total_throughput_gbps / num_days
        
        
        # Calculate power consumption metrics
        def transform_to_real_power_kwatts(num_pof_pools: int, power_series):
            ptx_on = 0.0136*30 + 27.34
            ptx_off = 27.34
            
            real_power_series = []
            max_femtos = num_pof_pools * 5
            
            for i in range(len(power_series)):
                p_i = power_series[i] # Watts
                # Calcular el número de femtos activos en el instante i
                # Cada femto consume 0.7W, por lo que el número de femtos activos es:
                num_femtos_active = p_i / 0.7
                num_femtos_active = int(round(num_femtos_active))
                
                power_this_series = ptx_on * num_femtos_active + ptx_off * (max_femtos - num_femtos_active)
                real_power_series.append(power_this_series/1000)
                
            return real_power_series

        power_comsumption_series = transform_to_real_power_kwatts(self.numberOfPofPools, self.live_smallcell_consumption)
        total_power_consumption_kWh = sum(power_comsumption_series) * self.timeStep/3600
        
        
        total_power_consumption_kWh = sum(power_comsumption_series) * self.timeStep/3600
        daily_avg_power_consumption_kWh = total_power_consumption_kWh / ((len(power_comsumption_series) - 1) * self.timeStep / (24*3600))
        yearly_power_estimate_kWh = daily_avg_power_consumption_kWh * 365
        
        # Calculate availability percentage
        # Calculate based on the number of users served by either femto or macro cells
        total_possible_connections = len(self.served_users_sim) * len(self.NUsers)
        total_nonserved_connections = np.sum(self.served_users_sim)  # Sum of all 1s (macro)
        availability_percentage = ((total_possible_connections - total_nonserved_connections) / total_possible_connections) * 100
        
        # Calculate blocked traffic
        # Sum up all blocked traffic across all users and time steps, convert to Gbps
        total_blocked_traffic_bps = np.sum(self.blocked_users_traffic_bps)
        blocked_traffic_gbps = total_blocked_traffic_bps / 1e9
        
        return {
            'total_throughput_gbps': total_throughput_gbps,
            'daily_avg_throughput_gbps': daily_avg_throughput_gbps,
            'total_power_consumption_kWh': total_power_consumption_kWh,
            'daily_avg_power_consumption_kWh': daily_avg_power_consumption_kWh,
            'yearly_power_estimate_kWh': yearly_power_estimate_kWh,
            'availability_percentage': availability_percentage,
            'blocked_traffic_gbps': blocked_traffic_gbps,
            'throughput_time_series_gbps': self.live_throughput / 1e9,
            'power_time_series_kWh': power_comsumption_series,
        }
