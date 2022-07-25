import numpy as np
from ant_colony_optimization.travelling_salesman_problem.data import Data
from ant_colony_optimization.travelling_salesman_problem.cost_function import travelling_salesman_problem
from ant_colony_optimization.travelling_salesman_problem.solution_methodology import ACO


def main():

    # data = Data(number_of_nodes=100)
    # data.create_and_save()
    model_data = Data.load()

    problem = travelling_salesman_problem

    solution_method = ACO(cost_function=problem,
                          max_iteration=100,
                          number_of_ants=100,
                          heuristic_information=np.divide(1, model_data["distances"]),
                          pheromone=np.ones_like(model_data["distances"]),
                          Q=2,
                          exponential_weight_for_heuristic=1,
                          exponential_weight_for_pheromone=3.8,
                          pheromone_evaporation_rate=0.04,
                          mutation=True,
                          plot=True)

    solution_best, run_time = solution_method.run()

    return solution_best, run_time


if __name__ == "__main__":

    solution, runtime = main()
