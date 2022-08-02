from ant_colony_optimization.travelling_salesman_problem.model_data import Data
from ant_colony_optimization.travelling_salesman_problem.problem import cost_function
from ant_colony_optimization.travelling_salesman_problem.ant_colony import ACO
import numpy as np


def main():

    # data = Data(number_of_nodes=50)
    # data.create_and_save()

    model_data = Data.load()

    distances = model_data["distances"]
    heuristics_information = np.divide(1, distances)
    pheromone = np.ones_like(heuristics_information)

    cost_func = cost_function

    solution_method = ACO(cost_function=cost_func,
                          max_iteration=100,
                          number_of_ants=150,
                          pheromone=pheromone,
                          pheromone_constant_update=1,
                          pheromone_exponent_rate=2.7,
                          pheromone_evaporation_rate=0.04,
                          heuristics_information=heuristics_information,
                          heuristics_exponent_rate=1.2,
                          plot_solution=True)

    solution_best, run_time = solution_method.run()

    return solution_best, run_time


if __name__ == "__main__":

    solution, runtime = main()
