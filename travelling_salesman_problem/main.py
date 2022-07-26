import numpy as np

from ant_colony_optimization.travelling_salesman_problem.data import Data
from ant_colony_optimization.travelling_salesman_problem.travelling_salesman_problem import cost_function
from ant_colony_optimization.travelling_salesman_problem.solution_method import ACO


def main():

    # data = Data(number_of_nodes=50)
    # data.create_and_save()

    model_data = Data.load()

    cost_func = cost_function

    solution_method = ACO(cost_function=cost_func,
                          max_iteration=200,
                          number_of_ants=200,
                          pheromone=np.ones_like(model_data["distances"]),
                          pheromone_evaporation_rate=0.02,
                          pheromone_constant=1,
                          pheromone_exponent_rate=2,
                          heuristic=np.divide(1, model_data["distances"]),
                          heuristic_exponent_rate=1,
                          mutation=True,
                          plot=True)

    solution_best, run_time = solution_method.run()

    return solution_best, run_time


if __name__ == "__main__":

    solution, runtime = main()
