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

        knapsack_capacity = np.reshape(np.sum(weights) / 5, (1, 1))

        os.makedirs("./data", exist_ok=True)
        np.save("./data/values.npy", values)
        np.save("./data/weights.npy", weights)
        np.save("./data/knapsack_capacity.npy", knapsack_capacity)

    @staticmethod
    def load():

        out = {

            "values": np.load("./data/values.npy"),
            "weights": np.load("./data/weights.npy"),
            "knapsack_capacity": np.load("./data/knapsack_capacity.npy")[0, 0]
        }

        return out
