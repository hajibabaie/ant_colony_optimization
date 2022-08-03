from ant_colony_optimization.quadratic_assignment_problem.model_data import Data
import matplotlib.pyplot as plt
import numpy as np
import time
import copy
import os


class ACO:

    class _Ant:

        def __init__(self):

            self.position = []
            self.cost = None

    def __init__(self,
                 cost_function,
                 max_iteration,
                 number_of_ants,
                 pheromone,
                 pheromone_constant,
                 pheromone_evaporation_rate,
                 pheromone_exponent_rate,
                 heuristic_information,
                 heuristic_exponent_rate,
                 plot_solution=False):

        self._cost_function = cost_function
        self._max_iteration = max_iteration
        self._number_of_ants = number_of_ants
        self._ants = None
        self._pheromone = pheromone
        self._pheromone_constant = pheromone_constant
        self._pheromone_evaporation_rate = pheromone_evaporation_rate
        self._pheromone_exponent = pheromone_exponent_rate
        self._heuristic_information = heuristic_information
        self._heuristic_exponent = heuristic_exponent_rate
        self._plot_solution = plot_solution
        self._best_ant = self._Ant()
        self._best_ant.cost = 1e20
        self._best_cost = []
        self._model_data = Data.load()

    @staticmethod
    def _roulette_wheel_selection(probs):

        random_number = np.random.rand()

        probs_cumsum = np.cumsum(probs)

        return int(np.argwhere(random_number < probs_cumsum)[0][0])

    def _initialize_ants(self):

        ants = [self._Ant() for _ in range(self._number_of_ants)]

        for i in range(self._number_of_ants):

            ants[i].position = [int(i) for i in np.random.choice(range(len(self._model_data["location_x"])), 1)]

            for k in range(len(self._model_data["location_x"]) - 1):

                last_position = ants[i].position[-1]

                P = np.power(self._pheromone[last_position, :], self._pheromone_exponent) * \
                    np.power(self._heuristic_information[last_position, :], self._heuristic_exponent)

                P[ants[i].position] = 0

                P /= np.sum(P)

                ants[i].position.append(self._roulette_wheel_selection(P))

        return ants

    def _evaluate_ants(self):

        for i in range(self._number_of_ants):

            self._ants[i].cost = self._cost_function(self._ants[i].position)

            if self._ants[i].cost < self._best_ant.cost:

                self._best_ant = copy.deepcopy(self._ants[i])

    def _update_pheromone(self):

        for i in range(self._number_of_ants):

            position = self._ants[i].position

            for k in range(len(position) - 1):

                self._pheromone[position[k], position[k + 1]] += self._pheromone_constant / self._ants[i].cost

                self._pheromone[position[k + 1], position[k]] = self._pheromone[position[k], position[k + 1]]

            self._pheromone[position[-1], position[0]] += self._pheromone_constant / self._ants[i].cost
            self._pheromone[position[0], position[-1]] = self._pheromone[position[-1], position[0]]

        self._pheromone *= (1 - self._pheromone_evaporation_rate)



    def run(self):

        tic = time.time()

        for iter_main in range(self._max_iteration):

            self._ants = self._initialize_ants()

            self._evaluate_ants()

            self._update_pheromone()

            self._best_cost.append(self._best_ant.cost)

        toc = time.time()

        if self._plot_solution:

            location_x = self._model_data["location_x"]
            location_y = self._model_data["location_y"]
            position = self._best_ant.position[:int(self._model_data["weight_between_facilities"].shape[0])]

            os.makedirs("./figures", exist_ok=True)

            plt.figure(dpi=300, figsize=(10, 6))

            plt.plot(range(self._max_iteration), self._best_cost)
            plt.title("Quadratic Assignment Problem Using Ant Colony Optimization", fontweight="bold")
            plt.xlabel("Number of Iteration")
            plt.ylabel("Cost")
            plt.savefig("./figures/cost_function.png")

            plt.figure(dpi=300, figsize=(10, 6))
            plt.scatter(location_x, location_y, marker="o", s=8, facecolor=None, edgecolors="red")
            for i in range(len(location_x)):
                plt.text(location_x[i] + 1,
                         location_y[i] - 1, str(i), c="red")
            plt.scatter(location_x[position], location_y[position], s=32, c="green")
            for i in range(len(position)):
                plt.text(location_x[position[i]], location_y[position[i]] + 0.5, str(i), fontweight="bold", c="green")

            plt.savefig("./figures/assignment.png")



        return self._best_ant, toc - tic
