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

            for j in range(i + 1, self._number_of_nodes):

                distances[i, j] = np.sqrt(np.square(location_x[i] - location_x[j]) +
                                          np.square(location_y[i] - location_y[j]))

                distances[j, i] = distances[i, j]

        os.makedirs("./data", exist_ok=True)
        np.savetxt("./data/location_x.csv", location_x, delimiter=",")
        np.savetxt("./data/location_y.csv", location_y, delimiter=",")
        np.savetxt("./data/distances.csv", distances, delimiter=",")

    @staticmethod
    def load():

        out = {
            "location_x": np.genfromtxt("./data/location_x.csv", delimiter=","),
            "location_y": np.genfromtxt("./data/location_y.csv", delimiter=","),
            "distances": np.genfromtxt("./data/distances.csv", delimiter=",")
        }

        return out
