import numpy as np
import os


class Data:

    def __init__(self,
                 number_of_possible_locations,
                 number_of_facilities):

        self._number_of_locations = number_of_possible_locations
        self._number_of_facilities = number_of_facilities

    def create_and_save(self):

        location_x = np.random.randint(0, 100, self._number_of_locations)
        location_y = np.random.randint(0, 100, self._number_of_locations)

        distances = np.zeros((self._number_of_locations, self._number_of_locations))

        for i in range(self._number_of_locations):

            for j in range(i + 1, self._number_of_locations):

                distances[i, j] = np.sqrt(np.square(location_x[i] - location_x[j]) +
                                          np.square(location_y[i] - location_y[j]))

                distances[j, i] = distances[i, j]

        weights_between_facilities = np.random.randint(10, 20, (self._number_of_facilities, self._number_of_facilities))

        weights_between_facilities -= np.diag(np.diag(weights_between_facilities))

        weights_between_facilities = np.divide(weights_between_facilities + weights_between_facilities.T, 2)

        os.makedirs("./data", exist_ok=True)
        np.savetxt("./data/location_x.csv", location_x, delimiter=",")
        np.savetxt("./data/location_y.csv", location_y, delimiter=",")
        np.savetxt("./data/distances.csv", distances, delimiter=",")
        np.savetxt("./data/weights_between_facilities.csv", weights_between_facilities, delimiter=",")

    @staticmethod
    def load():

        out = {
            "location_x": np.genfromtxt("./data/location_x.csv", delimiter=","),
            "location_y": np.genfromtxt("./data/location_y.csv", delimiter=","),
            "distances": np.genfromtxt("./data/distances.csv", delimiter=","),
            "weight_between_facilities": np.genfromtxt("./data/weights_between_facilities.csv", delimiter=",")
        }

        return out
