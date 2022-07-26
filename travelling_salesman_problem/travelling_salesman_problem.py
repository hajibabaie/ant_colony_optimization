from ant_colony_optimization.travelling_salesman_problem.data import Data


def cost_function(path):

    model_data = Data.load()
    distances = model_data["distances"]

    out = 0
    for i in range(len(path) - 1):

        out += distances[path[i], path[i + 1]]

    out += distances[path[-1], path[0]]

    return out
