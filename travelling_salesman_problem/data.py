import numpy as np
import os


class Data:

    def __init__(self, number_of_nodes):

        self._number_of_nodes = number_of_nodes

    def create_and_save(self):

        location_x = np.random.randint(0, 100, self._number_of_nodes)
        location_y = np.random.randint(0, 100, self._number_of_nodes)

        distances = np.zeros((self._number_of_nodes, self._number_of_nodes))

        for i in range(self._number_of_nodes):
            for j in range(self._number_of_nodes):

                distances[i, j] = np.sqrt(np.square(location_x[i] - location_x[j]) +
                                          np.square(location_y[i] - location_y[j]))

                distances[j, i] = distances[i, j]


        os.makedirs("./data", exist_ok=True)
        np.save("./data/location_x.npy", location_x)
        np.save("./data/location_y.npy", location_y)
        np.save("./data/distances.npy", distances)

    @staticmethod
    def load():

        out = {
            "location_x": np.load("./data/location_x.npy"),
            "location_y": np.load("./data/location_y.npy"),
            "distances": np.load("./data/distances.npy")
        }

        return out
