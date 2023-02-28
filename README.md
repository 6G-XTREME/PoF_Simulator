# PoF_Simulation_PYTHON

This repository contains the alternative code to the PoF Simulation using Python. The present repository includes:

* main.py: The main file in this repository. Please execute it to start the simulation
* main_no_plots.py: This script is very similar to main.py, but it does not include plots
* map_utils.py: This file includes the needed functions to operate with the maps
	* apollonius_circle_path_loss
	* get_circle
	* get_dominance_area
	* get euclidean_distance
	* perpendicular_bisector
	* search_closest_bs
* mobility_utils.py: It has the functions to define the mobility patterns in the simulation
	* Out_adjustDuration_random_waypoint
	* add_element_to_s_mobility
	* Out_setRestrictedWalk_random_waypoint
	* generate_mobility
* plot_utils.py: Includes only one functions to plot (TBD)
	* ema:
* polygon_cut.py: This file contains different functions to operate with shapely polygons
	* clip
	* inside
	* polyclip
* radio_utils.py: Different operations with radio in Macro and Femto cells
	* compute_sinr_dl
	* compute_sinr_ul
* requirements.txt: This file includes all the required dependencies to install in this environment
* user_association_utils.py: The included function find the closest macro cell
	* search_closest_macro