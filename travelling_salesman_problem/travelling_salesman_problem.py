from ant_colony_optimization.travelling_salesman_problem.data import Data


def cost_function(tour):

    distances = Data.load()["distances"]

    cost = 0
    for i in range(len(tour) - 1):

        cost += distances[tour[i], tour[i + 1]]

    cost += distances[tour[-1], tour[0]]

    return cost
