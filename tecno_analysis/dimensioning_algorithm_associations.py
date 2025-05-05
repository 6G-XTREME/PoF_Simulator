import numpy as np
import tecno_analysis.utility as Utils
import time
from math import ceil


from model.RegionsCalcs import create_regions, create_regions_overlapping
from shapely.geometry import Polygon
from numpy.typing import NDArray
from typing import Optional





class NodePositionsAlgorithm:
    
    
    def __init__(
        self,
        nodes_with_femtos: NDArray[np.int_],
        nodes_adjacencies_matrix: NDArray[np.float64],
        nodes_fixed_hplds: NDArray[np.int_],
        nodes_tentative_hplds: NDArray[np.int_],
    ):
        self.nodes_with_femtos = nodes_with_femtos
        self.nodes_adjacencies_matrix = nodes_adjacencies_matrix
        self.dijsktra_matrix = Utils.dijkstra_distances_matrix(nodes_adjacencies_matrix)
        self.nodes_fixed_hplds = nodes_fixed_hplds
        self.nodes_tentative_hplds = nodes_tentative_hplds
        
        
        fixed_femtos = sum(nodes_fixed_hplds) * 5
        free_femtos = sum(nodes_with_femtos) - fixed_femtos
        self.estimated_num_hplds = ceil(free_femtos / 5)
    


    def determine_best_hpld_positions(
        self,
        max_runtime: float = 15,
    ) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
        # Start with fixed HPLD positions
        initial_hpld_positions = self.nodes_fixed_hplds.copy()
        
        
        # Add random non-fixed HPLDs to initial solution
        rand_num_hplds = np.random.randint(max(1, len(self.nodes_tentative_hplds) // 4), len(self.nodes_tentative_hplds) + 1)
        random_nodes_indices = np.random.choice(self.nodes_tentative_hplds, rand_num_hplds, replace=False)
        initial_hpld_positions[random_nodes_indices] = 1
        
        # Get initial associations based on initial HPLD positions
        initial_associations = self.determine_best_associations(initial_hpld_positions)
        
        best_hpld_positions = initial_hpld_positions
        best_associations = initial_associations
        best_cost = self.get_augmented_cost(best_associations, best_hpld_positions)
        
        num_loops_no_improvement = 0
        num_loops_max = 20
        count_iterations = 0
        start_time = time.time()
        heuristic_1_evolution = [] # (iteration, cost, best cost, time)
        
        while num_loops_no_improvement < num_loops_max and (time.time() - start_time) < max_runtime:
            this_loop_cost = best_cost
            
            # Randomize the order of non-fixed HPLD candidate positions for this iteration
            shuffled_indices = np.random.permutation(self.nodes_tentative_hplds)
            
            for node in shuffled_indices:
                # Try toggling this non-fixed HPLD position
                local_hpld_positions = best_hpld_positions.copy()
                local_hpld_positions[node] = 1 - local_hpld_positions[node]  # Toggle. 1 -> 0 or 0 -> 1
                
                # Get new associations based on new HPLD positions
                local_associations = self.determine_best_associations(local_hpld_positions)
                
                new_cost = self.get_augmented_cost(local_associations, local_hpld_positions)
                
                count_iterations += 1
                heuristic_1_evolution.append((count_iterations, new_cost, best_cost, time.time() - start_time))
                
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_hpld_positions = local_hpld_positions
                    best_associations = local_associations
                    num_loops_no_improvement = 0
                
                if (time.time() - start_time) > max_runtime:
                    break
            
            if best_cost >= this_loop_cost:
                num_loops_no_improvement += 1
                
            if best_cost < this_loop_cost:
                num_loops_no_improvement = 0
            else:
                num_loops_no_improvement += 1
                    
        return best_hpld_positions, best_associations
                
        
    
    
    
    def determine_best_associations(self, hpld_positions: NDArray[np.int_], max_runtime: float = 15):
        # Empty solution
        nodes_hplds_to_femtos = np.zeros((len(self.nodes_with_femtos), len(self.nodes_with_femtos)), dtype=int)

        # Get indices of femtocells and HPLDs
        femto_indices = np.where(self.nodes_with_femtos == 1)[0]
        hpld_indices = np.where(hpld_positions == 1)[0]

        if len(hpld_indices) == 0 or len(femto_indices) == 0:
            return nodes_hplds_to_femtos  # Return empty solution if no HPLDs or femtocells

        # Calculate distances between HPLDs and femtocells
        distances = np.zeros((len(hpld_indices), len(femto_indices)))
        for i, hpld_idx in enumerate(hpld_indices):
            for j, femto_idx in enumerate(femto_indices):
                distances[i, j] = self.dijsktra_matrix[hpld_idx, femto_idx]

        # Assign femtocells to HPLDs (up to 5 per HPLD)
        assigned_femtos = set()
        for i, hpld_idx in enumerate(hpld_indices):
            # Get the 5 closest unassigned femtocells
            femto_distances = distances[i, :]
            closest_femtos = np.argsort(femto_distances)
            assigned_count = 0
            for femto_idx in closest_femtos:
                if femto_idx not in assigned_femtos and assigned_count < 5:
                    nodes_hplds_to_femtos[hpld_idx, femto_indices[femto_idx]] = 1
                    assigned_femtos.add(femto_idx)
                    assigned_count += 1

        return nodes_hplds_to_femtos




    def get_cost(self, hpld_associations: NDArray[np.int_], hpld_positions: NDArray[np.float64]):
        # Sum the distances between HPLDs and their associated femtocells
        total_distance = 0
        for i in range(hpld_associations.shape[0]):
            for j in range(hpld_associations.shape[0]):
                if hpld_associations[i, j] == 1:
                    distance = self.nodes_adjacencies_matrix[i, j]
                    total_distance += distance
        
        return total_distance
    
    
    
    def get_augmented_cost(self, hpld_associations: NDArray[np.int_], hpld_positions: NDArray[np.float64], penalization_cost_factor: float = 1000):
        # Penalize if any femto is not associated to any HPLD
        penalization_cost = penalization_cost_factor * (np.sum(hpld_associations) == 0)
        
        # Penalize if any HPLD is not associated to any femto
        penalization_cost += penalization_cost_factor * (np.sum(hpld_associations, axis=1) == 0).sum()
        
        # Penalize if any HPLD is associated to more than 5 femtos
        penalization_cost += penalization_cost_factor * (np.sum(hpld_associations, axis=1) > 5).sum()
        
        # Penalize if any femto is associated to more than 1 HPLD
        penalization_cost += penalization_cost_factor * (np.sum(hpld_associations, axis=0) > 1).sum()

        # Penalize if the number of HPLDs is greater than the estimated number of HPLDs
        penalization_cost += penalization_cost_factor * (np.sum(hpld_positions) > self.estimated_num_hplds)
        
        
        return self.get_cost(hpld_associations, hpld_positions) + penalization_cost