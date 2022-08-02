from ant_colony_optimization.binary_knapsack_problem.model_data import Data
import numpy as np


def cost_function(x):

    x = np.array(x)

    model_data = Data.load()
    values = model_data["values"]
    weights = model_data["weights"]
    knapsack_capacity = model_data["knapsack_capacity"]

    value_gained = np.dot(x, values)
    value_not_gained = np.dot(1 - x, values)

    weight_gained = np.dot(x, weights)
    weight_not_gained = np.dot(1 - x, weights)

    violation = np.maximum((weight_gained / knapsack_capacity) - 1, 0)

    cost = value_not_gained * (1 + 10 * violation)

    out = {
        "value_gained": value_gained,
        "value_not_gained": value_not_gained,
        "knapsack_capacity": knapsack_capacity,
        "weight_gained": weight_gained,
        "weight_not_gained": weight_not_gained,
        "violation": violation
    }

    return out, float(cost)
