import numpy as np
import tecno_analysis.utility as Utils
import time
from math import ceil


from model.RegionsCalcs import create_regions, create_regions_overlapping
from shapely.geometry import Polygon
from numpy.typing import NDArray
from typing import Optional


class DimensioningAlgorithmTecnoeconomics:
    def __init__(
        self,
        node_adjacencies_matrix: NDArray[np.float64],
        node_positions: NDArray[np.float64],
        node_type_hl4: NDArray[np.int_],
        node_type_hl5: NDArray[np.int_],
        node_traffic_injection: NDArray[np.float64],
        fixed_nodes_for_hpld: NDArray[np.int_],
        tentative_nodes_for_hpld: NDArray[np.int_],
        tentative_nodes_for_femtocells: NDArray[np.int_],
        tentative_range_for_femtocells: NDArray[np.float64],
        nodes_for_macrocells: NDArray[np.int_],
        range_for_macrocells: NDArray[np.float64],
        alpha_loss: float = 3,
        max_runtime_seconds: float = 180.0,
        euclidean_to_km_scale: float = 1.0,
        power_for_hpld: Optional[NDArray[np.float64]] = None,
        power_for_femtocells: Optional[NDArray[np.float64]] = None,
        power_for_macrocells: Optional[NDArray[np.float64]] = None,
        base_area: list[tuple[float, float]] = None,
        
        alpha: float = 0.31,
    ):
        self.node_adjacencies_matrix = node_adjacencies_matrix
        self.node_adjacencies_matrix_dijkstra = Utils.dijkstra_distances_matrix(node_adjacencies_matrix)
        self.node_positions = node_positions
        self.node_type_hl4 = node_type_hl4
        self.node_type_hl5 = node_type_hl5
        self.node_traffic_injection = node_traffic_injection
        self.fixed_nodes_for_hpld = fixed_nodes_for_hpld
        self.tentative_nodes_for_hpld = tentative_nodes_for_hpld
        self.tentative_nodes_for_femtocells = tentative_nodes_for_femtocells
        self.tentative_range_for_femtocells = tentative_range_for_femtocells
        self.nodes_for_macrocells = nodes_for_macrocells
        self.range_for_macrocells = range_for_macrocells
        self.max_runtime_seconds = max_runtime_seconds
        self.euclidean_to_km_scale = euclidean_to_km_scale
        self.power_for_hpld = power_for_hpld
        self.power_for_femtocells = power_for_femtocells
        self.power_for_macrocells = power_for_macrocells
        self.alpha_loss = alpha_loss
        self.alpha = alpha
        self.base_area = base_area
        self.bss = np.array(
            [
                [node_positions[i][0], node_positions[i][1], self.power_for_femtocells[i]]
                for i in range(len(node_positions))
            ]
        )

        self.nodes_with_hpld = np.zeros(len(node_adjacencies_matrix), dtype=int)
        self.nodes_with_femtocells = np.zeros(len(node_adjacencies_matrix), dtype=int)
        self.hpld_to_femtocell_association = np.zeros((len(node_adjacencies_matrix), len(node_adjacencies_matrix)), dtype=int)
        self.heuristic_1_evolution = []
        self.heuristic_2_evolution = []

    def run_algorithm(
        self,
        max_runtime_seconds: float = 180.0,
        time_division_between_phases: float = 0.5,
        alpha: float = 0.31,     # Balance between cost of HPLDs and femtocells. Recommended value: femto c.u. / hpld c.u.
        area_cost_magnifier: float = 1.0,
        throughput_cost_magnifier: float = 30.0,
        num_femtos_cost_magnifier: float = 1.0,
        penalization_cost: float = 100.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        
     
        # Some previous definitions
        num_nodes = len(self.node_positions)
        num_fixed_hplds = sum(self.fixed_nodes_for_hpld)
        num_tentative_hplds = sum(self.tentative_nodes_for_hpld)
        num_tentative_femtos = np.sum(self.tentative_nodes_for_femtocells)
        
        candidate_femtos_indices = np.where(self.tentative_nodes_for_femtocells == 1)[0]
        
        
        
        
        # 1. Dimension the location of the femtocells
        # 2. Dimension the location of the HPLDs based on the distance to the femtocells
        
        
        
        max_time_heuristic_1 = max_runtime_seconds * time_division_between_phases
        max_time_heuristic_2 = max_runtime_seconds * (1 - time_division_between_phases)
        
        
        # ----------------------------------------------------------------------------------------------------- #
        # -- Heuristic 1: Greedy approach for femtocells dimensioning ----------------------------------------- #
        #                                                                                                       #
        # ----------------------------------------------------------------------------------------------------- #
        # 1. Dimension the location of the femtocells
        start_time_heuristic_1 = time.time()
        heuristic_1_evolution = [] # (iteration, cost, best cost, time)
    
        # Random initial solution
        initial_solution = np.zeros(num_nodes, dtype=int)
        rand_num_femtos = np.random.randint(max(1, num_tentative_femtos // 4), num_tentative_femtos + 1)
        random_nodes_indices = np.random.choice(candidate_femtos_indices, rand_num_femtos, replace=False)
        initial_solution[random_nodes_indices] = 1
    
        best_solution = initial_solution
        best_cost = self.augmented_cost_femtocell(
            self.bss,
            self.power_for_femtocells,
            self.tentative_range_for_femtocells,
            self.node_traffic_injection,
            self.base_area,
            best_solution,
            alpha=alpha,
            num_hplds=num_fixed_hplds,
            area_cost_magnifier=area_cost_magnifier,
            throughput_cost_magnifier=throughput_cost_magnifier,
            num_femtos_cost_magnifier=num_femtos_cost_magnifier,
            penalization_cost=penalization_cost,
        )
    
        num_loops_no_improvement = 0
        num_loops_max = 20
        
        count_iterations = 0
        
        
        while num_loops_no_improvement < num_loops_max and (time.time() - start_time_heuristic_1) < max_time_heuristic_1:
            
            # Local Search Best Fit -> Evaluate all the neighbors and obtain the best one
            this_loop_cost = best_cost
    
            # Randomize the order of candidate indices for this iteration
            shuffled_indices = np.random.permutation(candidate_femtos_indices)
            for node in shuffled_indices:
                local_solution = best_solution.copy()
                local_solution[node] = 1 - local_solution[node]  # Toggle. 1 -> 0 or 0 -> 1
    
                new_cost = self.augmented_cost_femtocell(
                    self.bss,
                    self.power_for_femtocells,
                    self.tentative_range_for_femtocells,
                    self.node_traffic_injection,
                    self.base_area,
                    local_solution,
                    num_hplds=num_fixed_hplds,
                    alpha_loss=self.alpha_loss,
                    area_cost_magnifier=area_cost_magnifier,
                    throughput_cost_magnifier=throughput_cost_magnifier,
                    num_femtos_cost_magnifier=num_femtos_cost_magnifier,
                    penalization_cost=penalization_cost,
                    alpha=alpha,
                )
                
                count_iterations += 1
                heuristic_1_evolution.append((count_iterations, new_cost, best_cost, time.time() - start_time_heuristic_1))
    
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_solution = local_solution
    
                if (time.time() - start_time_heuristic_1) > max_time_heuristic_1:
                    break
                
            if best_cost < this_loop_cost:
                num_loops_no_improvement = 0
            else:
                num_loops_no_improvement += 1
                
            # heuristic_1_evolution.append((time.time() - start_time_heuristic_1, best_cost, count_iterations))
            
        self.nodes_with_femtocells = best_solution
        
    
        # while (time.time() - start_time_heuristic_1) < max_time_heuristic_1:
            
        #     # Random solution
        #     solution = np.zeros(num_nodes, dtype=int)
        #     num_hplds = num_fixed_hplds + np.random.randint(0, max(0, ceil((num_tentative_femtos - 5 * num_tentative_hplds) / 5)))
        #     num_max_femtos = num_hplds * 5
            
        #     rand_num_femtos = np.random.randint(num_max_femtos // 4, num_max_femtos + 1)
        #     random_nodes_indices = np.random.choice(candidate_femtos_indices, rand_num_femtos, replace=False)
        #     solution[random_nodes_indices] = 1
            
        #     # Optimize the solution
        #     while num_loops_no_improvement < num_loops_max:
        #         this_loop_cost = best_cost
        #         this_loop_solution = solution.copy()
                
                
        
        #         # Randomize the order of candidate indices for this iteration
        #         shuffled_indices = np.random.permutation(candidate_femtos_indices)
        #         for node in shuffled_indices:
        #             new_solution = this_loop_solution.copy()
        #             new_solution[node] = 1 - new_solution[node]  # Toggle. 1 -> 0 or 0 -> 1
        
        #             new_cost = self.augmented_cost_femtocell(
        #                 self.bss,
        #                 self.power_for_femtocells,
        #                 self.tentative_range_for_femtocells,
        #                 self.node_traffic_injection,
        #                 self.base_area,
        #                 new_solution,
        #                 alpha=alpha,
        #                 num_hplds=num_hplds,
        #                 area_cost_magnifier=area_cost_magnifier,
        #                 throughput_cost_magnifier=throughput_cost_magnifier,
        #                 num_femtos_cost_magnifier=num_femtos_cost_magnifier,
        #                 penalization_cost=penalization_cost,
        #             )
                    
        #             count_iterations += 1
        #             heuristic_1_evolution.append((count_iterations, new_cost, best_cost, time.time() - start_time_heuristic_1))
        
        #             if new_cost < best_cost:
        #                 best_cost = new_cost
        #                 best_solution = this_loop_solution
                        
        #             if new_cost < this_loop_cost:
        #                 this_loop_cost = new_cost
        #                 this_loop_solution = new_solution
        
        #             if (time.time() - start_time_heuristic_1) > max_time_heuristic_1:
        #                 break
                    
        #         if best_cost < this_loop_cost:
        #             num_loops_no_improvement = 0
        #         else:
        #             num_loops_no_improvement += 1
                    
        #         # heuristic_1_evolution.append((time.time() - start_time_heuristic_1, best_cost, count_iterations))
    
        # return best_solution
        
        
        
        
        
        
        # ----------------------------------------------------------------------------------------------------- #
        # -- Heuristic 2: Greedy approach for HPLDs dimensioning ---------------------------------------------- #
        #                                                                                                       #
        # ----------------------------------------------------------------------------------------------------- #
        start_time_heuristic_2 = time.time()
        heuristic_2_evolution = [] # (iteration, cost, best cost, time)
        
        # Get fixed HPLD indices
        fixed_hpld_indices = np.where(self.fixed_nodes_for_hpld == 1)[0]
        
        # Get tentative HPLD candidate indices
        candidate_hpld_indices = np.where(self.tentative_nodes_for_hpld == 1)[0]
        
        # Get femtocell indices from heuristic 1 solution
        femto_indices = np.where(self.nodes_with_femtocells == 1)[0]
        
        # Initial solution: fixed HPLDs + random tentative HPLDs
        initial_solution = self.fixed_nodes_for_hpld.copy()
        num_tentative_hplds = min(len(candidate_hpld_indices), 
                                 max(0, ceil((len(femto_indices) - 5 * len(fixed_hpld_indices)) / 5)))
        random_hplds = np.random.choice(candidate_hpld_indices, num_tentative_hplds, replace=False)
        initial_solution[random_hplds] = 1
        
        # Evaluate initial solution
        best_solution_hplds = initial_solution
        best_cost_hplds, best_association_hplds = self.augmented_cost_hplds(
            self.node_adjacencies_matrix,
            self.node_adjacencies_matrix_dijkstra,
            self.node_positions,
            self.nodes_with_femtocells,
            self.node_positions,
            best_solution_hplds,
            num_hplds=len(fixed_hpld_indices) + num_tentative_hplds,
            fixed_hpld_indices=fixed_hpld_indices,
            penalization_cost=penalization_cost,
        )
        
        num_loops_no_improvement = 0
        num_loops_max = 20
        count_iterations = 0
        
        
        # Local search optimization
        while num_loops_no_improvement < num_loops_max and (time.time() - start_time_heuristic_2) < max_time_heuristic_2:
            this_loop_cost = best_cost
            
            # inner_loop_cost = best_cost_hplds
            # while inner_loop_cost < this_loop_cost:
                # 
            #    pass 
                
                
                
                
                
                
                
                
            
            # Randomize the order of candidate indices for this iteration
            shuffled_indices = np.random.permutation(candidate_hpld_indices)
            for node in shuffled_indices:
                local_solution = best_solution.copy()
                local_solution[node] = 1 - local_solution[node]  # Toggle. 1 -> 0 or 0 -> 1
                
                # Evaluate local solution
                new_cost, new_association = self.augmented_cost_hplds(
                    self.node_adjacencies_matrix,
                    self.node_adjacencies_matrix_dijkstra,
                    self.node_positions,
                    self.nodes_with_femtocells,
                    self.node_positions,
                    local_solution,
                    num_hplds=np.sum(local_solution),
                    fixed_hpld_indices=fixed_hpld_indices,
                    penalization_cost=penalization_cost,
                )
                
                count_iterations += 1
                heuristic_2_evolution.append((count_iterations, new_cost, best_cost, time.time() - start_time_heuristic_2))
                
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_solution = local_solution
                    best_association_hplds = new_association
                
                if (time.time() - start_time_heuristic_2) > max_time_heuristic_2:
                    break
            
            if best_cost < this_loop_cost:
                num_loops_no_improvement = 0
            else:
                num_loops_no_improvement += 1
        
        
        
        
        # Save and return the results
        self.nodes_with_femtocells = best_solution
        self.nodes_with_hpld = best_solution_hplds
        self.hpld_to_femtocell_association = best_association_hplds
        self.heuristic_1_evolution = heuristic_1_evolution
        self.heuristic_2_evolution = heuristic_2_evolution
        
        return self.nodes_with_hpld, self.nodes_with_femtocells, self.hpld_to_femtocell_association, self.heuristic_1_evolution, self.heuristic_2_evolution

        
    @staticmethod
    def augmented_cost_femtocell(
        node_positions: NDArray[np.float64],
        node_power: NDArray[np.float64],
        tentative_range_for_femtocells: NDArray[np.float64],
        traffic_injection: NDArray[np.float64],
        base_area: list[tuple[float, float]],
        evaluating_solution: NDArray[np.int_],
        num_hplds: int,
        alpha_loss: float = 3,
        area_cost_magnifier: float = 1.0,
        throughput_cost_magnifier: float = 30.0,
        num_femtos_cost_magnifier: float = 1.0,
        penalization_cost: float = 100.0,
        alpha: float = 0.31,    # Balance between cost of HPLDs and femtocells. Recommended value: femto c.u. / hpld c.u.
    ) -> float:
    
        # total_nodes_for_femtos = np.sum(evaluating_solution)
        femtos_positions_and_power = []
        for i in range(len(node_positions)):
            if evaluating_solution[i] == 1:
                femtos_positions_and_power.append([node_positions[i][0], node_positions[i][1], node_power[i]])
                
        femtos_ranges = tentative_range_for_femtocells[evaluating_solution == 1]
    
        regions, _unsold_region = create_regions_overlapping(
            BaseStations=femtos_positions_and_power,
            alpha_loss=alpha_loss,
            polygon_bounds=base_area,
            max_radius_km_list=femtos_ranges,
        )
    
        # Calculate the covered area
        complete_area = Polygon(base_area)
        covered_area = (complete_area.area - _unsold_region.area)
        covered_area_percentage = covered_area / complete_area.area
    
        # Calculate the total throughput
        maximum_throughput = traffic_injection.sum()
        total_throughput = traffic_injection[evaluating_solution == 1].sum()
        throughput_percentage = total_throughput / maximum_throughput
        
        
        # Calculate the number of femtos
        num_femtos = np.sum(evaluating_solution)
        num_femtos_percentage = num_femtos / len(evaluating_solution)
    
    
    
        # Calculate the augmented cost function
        base_cost = (
            area_cost_magnifier / covered_area_percentage +
            throughput_cost_magnifier / throughput_percentage +
            num_femtos_cost_magnifier / num_femtos_percentage
        )
        
        # Penalizations
        # Penalize if the solution is empty
        # Penalize if the number of femtos exceeds the capacity
        penalizations = 0
        penalizations += penalization_cost * (num_femtos == 0)
        penalizations += penalization_cost * max(0, num_femtos - 5 * num_hplds)
        
        return base_cost + penalizations
    
    
    @staticmethod
    def augmented_cost_hplds(
        adjacency_matrix: NDArray[np.float64],
        dijkstra_distances_matrix: NDArray[np.float64],
        node_positions: NDArray[np.float64],
        femtocells_positions: NDArray[np.float64],
        hplds_positions: NDArray[np.float64],
        evaluating_solution: NDArray[np.int_],
        num_hplds: int,
        fixed_hpld_indices: NDArray[np.int_],
        penalization_cost: float = 100.0,
    ) -> tuple[float, NDArray[np.int_]]:
        """
        Evaluate the cost of a HPLD solution and return the cost and the association matrix.
        
        Args:
            adjacency_matrix: Adjacency matrix of the network
            dijkstra_distances_matrix: Dijkstra distances between all nodes
            node_positions: Positions of all nodes
            femtocells_positions: Binary array indicating which nodes have femtocells
            hplds_positions: Binary array indicating which nodes have HPLDs
            evaluating_solution: Binary array indicating which nodes are selected as HPLDs
            num_hplds: Total number of HPLDs (fixed + tentative)
            fixed_hpld_indices: Indices of fixed HPLDs
            penalization_cost: Cost to penalize unassigned femtocells
            
        Returns:
            tuple[float, NDArray[np.int_]]: (total cost, HPLD to femtocell association matrix)
        """
        # Get indices of HPLDs and femtocells
        hpld_indices = np.where(evaluating_solution == 1)[0]
        femto_indices = np.where(femtocells_positions == 1)[0]
        
        if len(hpld_indices) == 0 or len(femto_indices) == 0:
            return float('inf'), np.zeros((len(node_positions), len(node_positions)), dtype=int)
            
        # Calculate distances between HPLDs and femtocells
        distances = np.zeros((len(hpld_indices), len(femto_indices)))
        for i, hpld_idx in enumerate(hpld_indices):
            for j, femto_idx in enumerate(femto_indices):
                distances[i, j] = dijkstra_distances_matrix[hpld_idx, femto_idx]
        
        # Initialize association matrix
        association_matrix = np.zeros((len(node_positions), len(node_positions)), dtype=int)
        
        # Assign femtocells to HPLDs (up to 5 per HPLD)
        total_cost = 0
        assigned_femtos = set()
        
        # First assign to fixed HPLDs
        for i, hpld_idx in enumerate(hpld_indices):
            if hpld_idx in fixed_hpld_indices:
                # Get the 5 closest unassigned femtocells
                femto_distances = distances[i, :]
                closest_femtos = np.argsort(femto_distances)[:5]
                for femto_idx in closest_femtos:
                    if femto_idx not in assigned_femtos:
                        total_cost += femto_distances[femto_idx]
                        association_matrix[hpld_idx, femto_indices[femto_idx]] = 1
                        assigned_femtos.add(femto_idx)
        
        # Then assign remaining femtocells to tentative HPLDs
        for i, hpld_idx in enumerate(hpld_indices):
            if hpld_idx not in fixed_hpld_indices:
                # Get the 5 closest unassigned femtocells
                femto_distances = distances[i, :]
                closest_femtos = np.argsort(femto_distances)[:5]
                for femto_idx in closest_femtos:
                    if femto_idx not in assigned_femtos:
                        total_cost += femto_distances[femto_idx]
                        association_matrix[hpld_idx, femto_indices[femto_idx]] = 1
                        assigned_femtos.add(femto_idx)
        
        # Penalize if not all femtocells are assigned
        unassigned_penalty = penalization_cost * (len(femto_indices) - len(assigned_femtos))
        
        return total_cost + unassigned_penalty, association_matrix



# # ----------------------------------------------------------------------------------------------------- #
# # -- Heuristic 1: Greedy approach for femtocells dimensioning ----------------------------------------- #
# #                                                                                                       #
# # ----------------------------------------------------------------------------------------------------- #
# def determine_best_femtos(
#     node_positions: NDArray[np.float64],
#     node_power: NDArray[np.float64],
#     tentative_nodes_for_femtocells: NDArray[np.int_],
#     tentative_range_for_femtocells: NDArray[np.float64],
#     traffic_injection: NDArray[np.float64],
#     base_area: list[tuple[float, float]],
#     max_runtime_seconds: float = 60.0,
# ) -> NDArray[np.int_]:
#     start_time = time.time()
#     num_nodes = len(node_positions)
#     num_tentative_femtos = np.sum(tentative_nodes_for_femtocells)

#     # Random initial solution
#     initial_solution = np.zeros(num_nodes, dtype=int)
#     num_femtos = np.random.randint(max(1, num_tentative_femtos // 4), num_tentative_femtos + 1)
#     candidate_indices = np.where(tentative_nodes_for_femtocells == 1)[0]
#     random_nodes = np.random.choice(candidate_indices, num_femtos, replace=False)
#     initial_solution[random_nodes] = 1

#     best_solution = initial_solution
#     best_cost = augmented_cost_function_femtocell(
#         node_positions,
#         node_power,
#         tentative_range_for_femtocells,
#         traffic_injection,
#         base_area,
#         best_solution,
#         alpha_loss=3,
#     )

#     num_loops_no_improvement = 0
#     num_loops_max = 100

#     while num_loops_no_improvement < num_loops_max and (time.time() - start_time) < max_runtime_seconds:
#         this_loop_cost = best_cost

#         # Randomize the order of candidate indices for this iteration
#         shuffled_indices = np.random.permutation(candidate_indices)
#         for node in shuffled_indices:
#             local_solution = best_solution.copy()
#             local_solution[node] = 1 - local_solution[node]  # Toggle. 1 -> 0 or 0 -> 1

#             new_cost = augmented_cost_function_femtocell(
#                 node_positions,
#                 node_power,
#                 tentative_range_for_femtocells,
#                 traffic_injection,
#                 base_area,
#                 local_solution,
#                 alpha_loss=3,
#             )

#             if new_cost < best_cost:
#                 best_cost = new_cost
#                 best_solution = local_solution

#             if (time.time() - start_time) > max_runtime_seconds:
#                 break
#         if best_cost < this_loop_cost:
#             num_loops_no_improvement = 0
#         else:
#             num_loops_no_improvement += 1

#     return best_solution





# # ----------------------------------------------------------------------------------------------------- #
# # -- Cost function for heuristic 1 -------------------------------------------------------------------- #
# #                                                                                                       #
# # ----------------------------------------------------------------------------------------------------- #
# def augmented_cost_function_femtocell(
#     node_positions: NDArray[np.float64],
#     node_power: NDArray[np.float64],
#     tentative_range_for_femtocells: NDArray[np.float64],
#     traffic_injection: NDArray[np.float64],
#     base_area: list[tuple[float, float]],
#     evaluating_solution: NDArray[np.int_],
#     alpha_loss: float = 3,
#     area_cost_magnifier: float = 1.0,
#     throughput_cost_magnifier: float = 30.0,
#     num_femtos_cost_magnifier: float = 1.0,
#     penalization_cost: float = 100.0,
# ) -> float:

#     # total_nodes_for_femtos = np.sum(evaluating_solution)
#     femtos_positions_and_power = []
#     for i in range(len(node_positions)):
#         if evaluating_solution[i] == 1:
#             femtos_positions_and_power.append([node_positions[i][0], node_positions[i][1], node_power[i]])
            
#     femtos_ranges = tentative_range_for_femtocells[evaluating_solution == 1]

#     regions, _unsold_region = create_regions_overlapping(
#         BaseStations=femtos_positions_and_power,
#         alpha_loss=alpha_loss,
#         polygon_bounds=base_area,
#         max_radius_km_list=femtos_ranges,
#     )

#     # Calculate the covered area
#     complete_area = Polygon(base_area)
#     covered_area = (complete_area.area - _unsold_region.area)
#     covered_area_percentage = covered_area / complete_area.area

#     # Calculate the total throughput
#     maximum_throughput = traffic_injection.sum()
#     total_throughput = traffic_injection[evaluating_solution == 1].sum()
#     throughput_percentage = total_throughput / maximum_throughput
    
    
#     # Calculate the number of femtos
#     num_femtos = np.sum(evaluating_solution)
#     num_femtos_percentage = num_femtos / len(evaluating_solution)



#     # Calculate the augmented cost function
#     base_cost = (
#         area_cost_magnifier / covered_area_percentage +
#         # overlap_cost_magnifier * total_overlap_area_percentage +
#         throughput_cost_magnifier / throughput_percentage +
#         num_femtos_cost_magnifier / num_femtos_percentage
#     )
    
#     # Penalize if the solution is empty
#     penalization_cost = penalization_cost * (num_femtos == 0)
    
#     return base_cost + penalization_cost




