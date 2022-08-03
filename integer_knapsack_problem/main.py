from ant_colony_optimization.integer_knapsack_problem.model_data import Data
from ant_colony_optimization.integer_knapsack_problem.problem import cost_function
from ant_colony_optimization.integer_knapsack_problem.ant_colony import ACO
import numpy as np


def main():

    # data = Data(number_of_items=30,
    #             min_range_of_quantity=5,
    #             max_range_of_quantity=10,
    #             min_range_of_values=100,
    #             max_range_of_values=500,
    #             min_range_of_weights=20,
    #             max_range_of_weights=70)
    #
    # data.create_and_save()
    model_data = Data.load()
    values = model_data["values"]
    weights = model_data["weights"]
    number_of_each_items = model_data["number_of_each_items"]

    heuristic_information = {k: values[k] / weights[k] * np.ones((1, int(number_of_each_items[k] + 1)))
                             for k in range(len(number_of_each_items))}

    pheromone = {k: np.ones_like(heuristic_information[k])
                 for k in range(len(number_of_each_items))}




    cost_func = cost_function

    solution_method = ACO(cost_function=cost_func,
                          max_iteration=100,
                          number_of_ants=120,
                          pheromone=pheromone,
                          pheromone_constant=1,
                          pheromone_evaporation=0.04,
                          pheromone_exponent=3.2,
                          heuristic_information=heuristic_information,
                          heuristic_exponent=1.2,
                          plot_solution=True)

    solution, run_time = solution_method.run()

    return solution, run_time


if __name__ == "__main__":


    solution_best, runtime = main()
