from ant_colony_optimization.sphere.problem import cost_function
from ant_colony_optimization.sphere.ant_colony import ACOR


def main():


    cost_func = cost_function

    type_number_of_variables = {"real1D": [5]}

    solution_method = ACOR(cost_function=cost_func,
                           type_number_of_variables=type_number_of_variables,
                           min_range_of_variables=-10,
                           max_range_of_variables=10,
                           max_iteration=1000,
                           number_of_ants=10,
                           number_of_new_samples=50,
                           intensification_factor=0.5,
                           deviation_rate=1,
                           plot_solution=True)

    solution_best, run_time = solution_method.run()

    return solution_best, run_time


if __name__ == "__main__":

    solution, runtime = main()
