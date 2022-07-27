from ant_colony_optimization.travelling_salesman_problem.data import Data
from ant_colony_optimization.travelling_salesman_problem.travelling_salesman_problem import cost_function
from ant_colony_optimization.travelling_salesman_problem.solution_methodology import ACO
import numpy as np


def main():

    # data = Data(number_of_nodes=50)
    # data.create_and_save()

    model_data = Data.load()

    heuristic_information = np.divide(1, model_data["distances"])
    pheromone = np.ones_like(heuristic_information)

    cost_func = cost_function

    solution_method = ACO(cost_function=cost_func,
                          max_iteration=500,
                          number_of_ants=100,
                          pheromone=pheromone,
                          pheromone_evaporation_rate=0.02,
                          pheromone_constant=1,
                          pheromone_exponent_rate=2,
                          heuristic=heuristic_information,
                          heuristic_exponent_rate=1,
                          plot=True)

    solution_best, run_time = solution_method.run()

    return solution_best, run_time


if __name__ == "__main__":

    solution, runtime = main()
