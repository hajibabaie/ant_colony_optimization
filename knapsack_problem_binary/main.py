from ant_colony_optimization.knapsack_problem_binary.data import Data
from ant_colony_optimization.knapsack_problem_binary.knapsack_problem import cost_function
from ant_colony_optimization.knapsack_problem_binary.solution_method import ACO
import numpy as np


def main():

    # data = Data(number_of_items=50,
    #             min_range_of_values=100,
    #             max_range_of_values=500,
    #             min_range_of_weights=20,
    #             max_range_of_weights=70)
    #
    # data.create_and_save()

    model_data = Data.load()

    cost_func = cost_function

    values = model_data["values"]
    weights = model_data["weights"]
    heuristic_information = np.zeros((2, len(values)))
    heuristic_information[0, :] = np.divide(values, weights)
    heuristic_information[1, :] = np.divide(values, weights)

    pheromone = np.ones_like(heuristic_information)

    solution = ACO(cost_function=cost_func,
                   max_iteration=500,
                   number_of_ants=100,
                   pheromone=pheromone,
                   pheromone_exponent_rate=1,
                   pheromone_evaporation_rate=0.04,
                   pheromone_constant_rate=1,
                   heuristic_exponent_rate=1,
                   heuristic_information=heuristic_information,
                   plot_cost=True)

    solution, runtime = solution.run()

    return solution, runtime


if __name__ == "__main__":

    solution, runtime = main()