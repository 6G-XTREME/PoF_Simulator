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
        
        nodes_with_hpld = np.zeros(len(self.node_positions), dtype=int)
        nodes_with_femtocells = np.zeros(len(self.node_positions), dtype=int)
        hpld_to_femtocell_association = np.zeros((len(self.node_positions), len(self.node_positions)), dtype=int)      
                
     
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
        
    
        # while (time.time() - start_time) < max_runtime_seconds:
            
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
        
        #             if new_cost < best_cost:
        #                 best_cost = new_cost
        #                 best_solution = this_loop_solution
                        
        #             if new_cost < this_loop_cost:
        #                 this_loop_cost = new_cost
        #                 this_loop_solution = new_solution
        
        #             if (time.time() - start_time) > max_runtime_seconds:
        #                 break
                    
        #         if best_cost < this_loop_cost:
        #             num_loops_no_improvement = 0
        #         else:
        #             num_loops_no_improvement += 1
                    
        #         heuristic_1_evolution.append((time.time() - start_time_heuristic_1, best_cost, count_iterations))
    
        # # return best_solution
        
        
        
        
        heuristic_2_evolution = [] # (iteration, cost, best cost, time)
        
        
        
        
        
        
        
        
        
        # Save and return the results
        self.nodes_with_femtocells = best_solution
        self.heuristic_1_evolution = heuristic_1_evolution
        
        
        # TODO: after HPLDs are dimensioned
        self.nodes_with_hpld = nodes_with_hpld
        self.hpld_to_femtocell_association = hpld_to_femtocell_association
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
        


# ----------------------------------------------------------------------------------------------------- #
# -- Heuristic 1: Greedy approach for femtocells dimensioning ----------------------------------------- #
#                                                                                                       #
# ----------------------------------------------------------------------------------------------------- #
def determine_best_femtos(
    node_positions: NDArray[np.float64],
    node_power: NDArray[np.float64],
    tentative_nodes_for_femtocells: NDArray[np.int_],
    tentative_range_for_femtocells: NDArray[np.float64],
    traffic_injection: NDArray[np.float64],
    base_area: list[tuple[float, float]],
    max_runtime_seconds: float = 60.0,
) -> NDArray[np.int_]:
    start_time = time.time()
    num_nodes = len(node_positions)
    num_tentative_femtos = np.sum(tentative_nodes_for_femtocells)

    # Random initial solution
    initial_solution = np.zeros(num_nodes, dtype=int)
    num_femtos = np.random.randint(max(1, num_tentative_femtos // 4), num_tentative_femtos + 1)
    candidate_indices = np.where(tentative_nodes_for_femtocells == 1)[0]
    random_nodes = np.random.choice(candidate_indices, num_femtos, replace=False)
    initial_solution[random_nodes] = 1

    best_solution = initial_solution
    best_cost = augmented_cost_function_femtocell(
        node_positions,
        node_power,
        tentative_range_for_femtocells,
        traffic_injection,
        base_area,
        best_solution,
        alpha_loss=3,
    )

    num_loops_no_improvement = 0
    num_loops_max = 100

    while num_loops_no_improvement < num_loops_max and (time.time() - start_time) < max_runtime_seconds:
        this_loop_cost = best_cost

        # Randomize the order of candidate indices for this iteration
        shuffled_indices = np.random.permutation(candidate_indices)
        for node in shuffled_indices:
            local_solution = best_solution.copy()
            local_solution[node] = 1 - local_solution[node]  # Toggle. 1 -> 0 or 0 -> 1

            new_cost = augmented_cost_function_femtocell(
                node_positions,
                node_power,
                tentative_range_for_femtocells,
                traffic_injection,
                base_area,
                local_solution,
                alpha_loss=3,
            )

            if new_cost < best_cost:
                best_cost = new_cost
                best_solution = local_solution

            if (time.time() - start_time) > max_runtime_seconds:
                break
        if best_cost < this_loop_cost:
            num_loops_no_improvement = 0
        else:
            num_loops_no_improvement += 1

    return best_solution





# ----------------------------------------------------------------------------------------------------- #
# -- Cost function for heuristic 1 -------------------------------------------------------------------- #
#                                                                                                       #
# ----------------------------------------------------------------------------------------------------- #
def augmented_cost_function_femtocell(
    node_positions: NDArray[np.float64],
    node_power: NDArray[np.float64],
    tentative_range_for_femtocells: NDArray[np.float64],
    traffic_injection: NDArray[np.float64],
    base_area: list[tuple[float, float]],
    evaluating_solution: NDArray[np.int_],
    alpha_loss: float = 3,
    area_cost_magnifier: float = 1.0,
    throughput_cost_magnifier: float = 30.0,
    num_femtos_cost_magnifier: float = 1.0,
    penalization_cost: float = 100.0,
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
        # overlap_cost_magnifier * total_overlap_area_percentage +
        throughput_cost_magnifier / throughput_percentage +
        num_femtos_cost_magnifier / num_femtos_percentage
    )
    
    # Penalize if the solution is empty
    penalization_cost = penalization_cost * (num_femtos == 0)
    
    return base_cost + penalization_cost




