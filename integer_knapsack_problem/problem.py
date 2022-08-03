from ant_colony_optimization.integer_knapsack_problem.model_data import Data
import numpy as np


def cost_function(x):

    model_data = Data.load()
    number_of_each_items = model_data["number_of_each_items"]
    values = model_data["values"]
    weights = model_data["weights"]
    knapsack_capacity = model_data["knapsack_capacity"]

    values_gained = np.dot(x, values)
    values_not_gained = np.dot(number_of_each_items - x, values)

    weights_gained = np.dot(x, weights)
    weights_not_gained = np.dot(number_of_each_items - x, weights)

    violation = np.maximum(np.divide(weights_gained, knapsack_capacity) - 1, 0)

    cost = values_not_gained * (1 + 10 * violation)

    out = {
        "values_gained": values_gained,
        "values_not_gained": values_not_gained,
        "weights_gained": weights_gained,
        "weights_not_gained": weights_not_gained,
        "violation": violation,
        "knapsack_capacity": knapsack_capacity
    }

    return out, cost
