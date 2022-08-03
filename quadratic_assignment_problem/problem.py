from ant_colony_optimization.quadratic_assignment_problem.model_data import Data


def cost_function(x):

    model_data = Data.load()

    distances = model_data["distances"]
    weights = model_data["weight_between_facilities"]

    cost = 0
    for i in range(int(weights.shape[0])):

        for j in range(i + 1, int(weights.shape[0])):

            cost += distances[x[i], x[j]] * weights[i, j]


    return cost



