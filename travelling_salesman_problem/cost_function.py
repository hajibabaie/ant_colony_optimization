from ant_colony_optimization.travelling_salesman_problem.data import Data


def travelling_salesman_problem(tour):

    model_data = Data.load()
    distances = model_data["distances"]

    cost = 0
    for i in range(len(tour) - 1):

        cost += distances[tour[i], tour[i + 1]]

    cost += distances[tour[-1], tour[0]]

    return cost
