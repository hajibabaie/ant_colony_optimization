from ant_colony_optimization.sphere.problem import cost_function
from ant_colony_optimization.sphere.ant_colony_real import ACOR


def main():

    cost_func = cost_function

    solution_method = ACOR(cost_function=cost_func,
                           min_range_of_variables=-10,
                           max_range_of_variables=10,
                           number_of_variables=5,
                           max_iteration=1000,
                           number_of_ants=20,
                           number_of_samples=50,
                           intensification_factor=0.5,
                           deviation_factor=1)

    solution_best, run_time = solution_method.run()

    return solution_best, run_time


if __name__ == "__main__":

    solution, runtime = main()
