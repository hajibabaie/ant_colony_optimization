import numpy as np
import os


class Data:

    def __init__(self,
                 number_of_items,
                 min_range_of_values,
                 max_range_of_values,
                 min_range_of_weights,
                 max_range_of_weights):

        self._number_of_items = number_of_items
        self._min_range_of_values = min_range_of_values
        self._max_range_of_values = max_range_of_values
        self._min_range_of_weights = min_range_of_weights
        self._max_range_of_weights = max_range_of_weights

    def create_and_save(self):

        values = np.random.randint(self._min_range_of_values,
                                   self._max_range_of_values,
                                   self._number_of_items)

        weights = np.random.randint(self._min_range_of_weights,
                                    self._max_range_of_weights,
                                    self._number_of_items)

        knapsack_capacity = np.sum(weights) / 5
        knapsack_capacity = knapsack_capacity.reshape(1, 1)

        os.makedirs("./data", exist_ok=True)
        np.savetxt("./data/values.csv", values, delimiter=",")
        np.savetxt("./data/weights.csv", weights, delimiter=",")
        np.savetxt("./data/knapsack_capacity.csv", knapsack_capacity, delimiter=",")

    @staticmethod
    def load():

        out = {
            "values": np.genfromtxt("./data/values.csv", delimiter=","),
            "weights": np.genfromtxt("./data/weights.csv", delimiter=","),
            "knapsack_capacity": float(np.genfromtxt("./data/knapsack_capacity.csv", delimiter=","))
        }

        return out
