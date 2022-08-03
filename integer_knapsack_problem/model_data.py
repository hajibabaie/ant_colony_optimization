import numpy as np
import os


class Data:

    def __init__(self,
                 number_of_items,
                 min_range_of_quantity,
                 max_range_of_quantity,
                 min_range_of_values,
                 max_range_of_values,
                 min_range_of_weights,
                 max_range_of_weights):

        self._number_of_items = number_of_items
        self._min_range_of_quantity = min_range_of_quantity
        self._max_range_of_quantity = max_range_of_quantity
        self._min_range_of_values = min_range_of_values
        self._max_range_of_values = max_range_of_values
        self._min_range_of_weights = min_range_of_weights
        self._max_range_of_weights = max_range_of_weights

    def create_and_save(self):

        number_of_each_items = np.random.randint(self._min_range_of_quantity,
                                                 self._max_range_of_quantity,
                                                 self._number_of_items)

        values = np.random.randint(self._min_range_of_values,
                                   self._max_range_of_values,
                                   self._number_of_items)

        weights = np.random.randint(self._min_range_of_weights,
                                    self._max_range_of_weights,
                                    self._number_of_items)

        knapsack_capacity = np.reshape(np.dot(number_of_each_items, weights) / 4, (1, 1))

        os.makedirs("./data", exist_ok=True)
        np.savetxt("./data/number_of_each_items.csv", number_of_each_items, delimiter=",")
        np.savetxt("./data/values.csv", values, delimiter=",")
        np.savetxt("./data/weights.csv", weights, delimiter=",")
        np.savetxt("./data/knapsack_capacity.csv", knapsack_capacity, delimiter=",")

    @staticmethod
    def load():

        out = {

            "number_of_each_items": np.genfromtxt("./data/number_of_each_items.csv", delimiter=","),
            "values": np.genfromtxt("./data/values.csv", delimiter=","),
            "weights": np.genfromtxt("./data/weights.csv", delimiter=","),
            "knapsack_capacity": float(np.genfromtxt("./data/knapsack_capacity.csv", delimiter=","))
        }

        return out
