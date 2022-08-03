from ant_colony_optimization.quadratic_assignment_problem.model_data import Data
from ant_colony_optimization.quadratic_assignment_problem.problem import cost_function
from ant_colony_optimization.quadratic_assignment_problem.ant_colony import ACO
import numpy as np


def main():

    # data = Data(number_of_possible_locations=20,
    #             number_of_facilities=10)
    # data.create_and_save()

    model_data = Data.load()
    distances = model_data["distances"]
    heuristic_information = np.divide(1, distances)
    pheromone = np.ones_like(heuristic_information)

    cost_func = cost_function

    solution_method = ACO(cost_function=cost_func,
                          max_iteration=100,
                          number_of_ants=40,
                          pheromone=pheromone,
                          pheromone_constant=1,
                          pheromone_exponent_rate=2.7,
                          pheromone_evaporation_rate=0.04,
                          heuristic_information=heuristic_information,
                          heuristic_exponent_rate=1.02,
                          plot_solution=True)

    solution_best, run_time = solution_method.run()

    return solution_best, run_time


if __name__ == "__main__":

    solution, runtime = main()
