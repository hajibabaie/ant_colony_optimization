from ant_colony_optimization.travelling_salesman_problem.model_data import Data
import matplotlib.pyplot as plt
import numpy as np
import copy
import time
import os


class ACO:

    class _Ant:

        def __init__(self):

            self.tour = []
            self.cost = None

    def __init__(self,
                 cost_function,
                 max_iteration,
                 number_of_ants,
                 pheromone,
                 pheromone_constant_update,
                 pheromone_exponent_rate,
                 pheromone_evaporation_rate,
                 heuristics_information,
                 heuristics_exponent_rate,
                 plot_solution=False):

        self._cost_function = cost_function
        self._max_iteration = max_iteration
        self._number_of_ants = number_of_ants
        self._ants = None
        self._pheromone = pheromone
        self._pheromone_constant = pheromone_constant_update
        self._pheromone_exponent_rate = pheromone_exponent_rate
        self._pheromone_evaporation = pheromone_evaporation_rate
        self._heuristics_information = heuristics_information
        self._heuristics_exponent_rate = heuristics_exponent_rate
        self._plot_solution = plot_solution
        self._best_ant = self._Ant()
        self._best_ant.cost = 1e20
        self._best_cost = []
        self._model_data = Data.load()

    @staticmethod
    def _roulette_wheel_selection(probs):

        random_number = np.random.random()

        probs_cumsum = np.cumsum(probs)

        return int(np.argwhere(random_number < probs_cumsum)[0][0])

    def _initialize_ants(self):

        ants = [self._Ant() for _ in range(self._number_of_ants)]

        for i in range(self._number_of_ants):

            ants[i].tour = [int(i) for i in np.random.choice(range(len(self._model_data["location_x"])), 1)]

            for k in range(len(self._model_data["location_x"]) - 1):

                last_visited_node = ants[i].tour[-1]

                P = np.power(self._pheromone[last_visited_node, :], self._pheromone_exponent_rate) * \
                    np.power(self._heuristics_information[last_visited_node, :], self._heuristics_exponent_rate)

                P[ants[i].tour] = 0

                P /= np.sum(P)

                ants[i].tour.append(self._roulette_wheel_selection(P))

        return ants

    def _evaluate_ants(self):

        for i in range(self._number_of_ants):

            self._ants[i].cost = self._cost_function(self._ants[i].tour)

            if self._ants[i].cost < self._best_ant.cost:

                self._best_ant = copy.deepcopy(self._ants[i])

                indices = [int(i) for i in np.random.choice(range(len(self._best_ant.tour)), 2, replace=False)]

                min_index, max_index = min(indices), max(indices)

                best_ant_tour = self._best_ant.tour
                new_tour = np.concatenate((best_ant_tour[:min_index],
                                           np.flip(best_ant_tour[min_index: max_index]),
                                           best_ant_tour[max_index:]))
                new_tour = [int(i) for i in new_tour]
                new_cost = self._cost_function(new_tour)

                if new_cost < self._best_ant.cost:

                    self._best_ant.tour = copy.copy(new_tour)
                    self._best_ant.cost = copy.copy(new_cost)

    def _update_pheromone(self):

        for i in range(self._number_of_ants):

            tour = self._ants[i].tour

            for k in range(len(tour) - 1):

                self._pheromone[tour[k], tour[k + 1]] += self._pheromone_constant / self._ants[i].cost

                self._pheromone[tour[k + 1], tour[k]] = self._pheromone[tour[k], tour[k + 1]]

            self._pheromone[tour[-1], tour[0]] += self._pheromone_constant / self._ants[i].cost

            self._pheromone[tour[0], tour[-1]] = self._pheromone[tour[-1], tour[0]]

        self._pheromone *= (1 - self._pheromone_evaporation)

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
            tour = self._best_ant.tour

            os.makedirs("./figures", exist_ok=True)

            plt.figure(dpi=300, figsize=(10, 6))
            plt.plot(range(self._max_iteration), self._best_cost)
            plt.xlabel("Number of Iteration")
            plt.ylabel("Cost")
            plt.title("Travelling Salesman Problem Using Ant Colony Optimization", fontweight="bold")
            plt.savefig("./figures/cost_function.png")

            plt.figure(dpi=300, figsize=(10, 6))

            plt.scatter(location_x, location_y, marker="o", s=8)
            for i in range(len(location_x)):
                plt.text(location_x[i], location_y[i], str(i))

            for i in range(len(tour) - 1):

                if i == 0:

                    plt.plot([location_x[tour[i]], location_x[tour[i + 1]]],
                             [location_y[tour[i]], location_y[tour[i + 1]]], color="green")

                else:

                    plt.plot([location_x[tour[i]], location_x[tour[i + 1]]],
                             [location_y[tour[i]], location_y[tour[i + 1]]], color="black")

            plt.plot([location_x[tour[-1]], location_x[tour[0]]],
                     [location_y[tour[-1]], location_y[tour[0]]], color="red")


            plt.savefig("./figures/tour.png")

        return self._best_ant, toc - tic
