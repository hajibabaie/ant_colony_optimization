from ant_colony_optimization.binary_knapsack_problem.model_data import Data
from ant_colony_optimization.binary_knapsack_problem.problem import cost_function
from ant_colony_optimization.binary_knapsack_problem.ant_colony import ACO
import numpy as np


def main():

    # data = Data(number_of_items=30,
    #             min_range_of_values=100,
    #             max_range_of_values=500,
    #             min_range_of_weights=20,
    #             max_range_of_weights=70)
    #
    # data.create_and_save()

    model_data = Data.load()

    values = model_data["values"]
    weights = model_data["weights"]
    heuristics_information = np.zeros((2, len(values)))
    heuristics_information[[0, 1], :] = np.divide(values, weights)
    pheromone = np.ones_like(heuristics_information)

    cost_func = cost_function

    solution_method = ACO(cost_function=cost_func,
                          max_iteration=100,
                          number_of_ants=200,
                          pheromone=pheromone,
                          pheromone_exponent_rate=2,
                          pheromone_evaporation_rate=0.04,
                          pheromone_constant=1,
                          heuristic_information=heuristics_information,
                          heuristic_exponent_rate=1.5,
                          plot_solution=True)

    solution_best, run_time = solution_method.run()

    return solution_best, run_time


if __name__ == "__main__":

    solution, runtime = main()
