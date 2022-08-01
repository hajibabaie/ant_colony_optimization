from ant_colony_optimization.travelling_salesman_problem.data import Data
from ant_colony_optimization.travelling_salesman_problem.travelling_salesman_problem import cost_function
from ant_colony_optimization.travelling_salesman_problem.solution_method import ACO
import numpy as np


def main():

    # data = Data(number_of_nodes=50)
    # data.create_and_save()
    model_data = Data.load()

    distances = model_data["distances"]
    heuristic_information = np.divide(1, distances)
    pheromone = np.ones_like(heuristic_information)

    cost_func = cost_function

    solution = ACO(cost_function=cost_func,
                   max_iteration=250,
                   number_of_ants=100,
                   pheromone=pheromone,
                   pheromone_constant_update=1,
                   pheromone_exponent_rate=2,
                   pheromone_evaporation_rate=0.03,
                   heuristic=heuristic_information,
                   heuristic_exponent_rate=1.7,
                   plot_solution=True)

    ants, runtime = solution.run()

    return ants, runtime


if __name__ == "__main__":

    ants, runtime = main()