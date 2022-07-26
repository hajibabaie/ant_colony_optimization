from ant_colony_optimization.travelling_salesman_problem.data import Data
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
                 pheromone_exponent_rate,
                 pheromone_evaporation_rate,
                 pheromone_constant,
                 heuristic,
                 heuristic_exponent_rate,
                 mutation,
                 plot):

        self._cost_function = cost_function
        self._max_iteration = max_iteration
        self._number_of_ants = number_of_ants
        self._ants = None
        self._pheromone = pheromone
        self._pheromone_exponent_rate = pheromone_exponent_rate
        self._pheromone_evaporation_rate = pheromone_evaporation_rate
        self._pheromone_constant = pheromone_constant
        self._heuristic = heuristic
        self._heuristic_exponent_rate = heuristic_exponent_rate
        self._best_ant = self._Ant()
        self._best_ant.cost = 1e20
        self._mutation = mutation
        self._plot = plot
        self._model_data = Data.load()
        self._best_cost = []
        self._new_ant = None

    @staticmethod
    def _roulette_wheel_selection(probs):

        random_number = np.random.random()
        probs_cum_sum = np.cumsum(probs)
        return int(np.argwhere(random_number < probs_cum_sum)[0][0])

    def _initialize_ants(self):

        ants = [self._Ant() for _ in range(self._number_of_ants)]

        for i in range(self._number_of_ants):

            ants[i].tour = [int(np.random.choice(range(len(self._model_data["location_x"])), 1))]

            for k in range(len(self._model_data["location_x"]) - 1):

                last_visited_node = ants[i].tour[-1]

                P = np.power(self._pheromone[last_visited_node, :], self._pheromone_exponent_rate) * \
                    np.power(self._heuristic[last_visited_node, :], self._heuristic_exponent_rate)

                P[ants[i].tour] = 0

                P /= np.sum(P)

                ants[i].tour.append(self._roulette_wheel_selection(P))

        return ants

    def _evaluation_ants(self, ants):

        for i in range(len(ants)):

            ants[i].cost = self._cost_function(ants[i].tour)

            if ants[i].cost < self._best_ant.cost:

                self._best_ant = copy.deepcopy(ants[i])

                if self._mutation:

                    self._new_ant = self._Ant()
                    self._new_ant.tour = self._apply_mutation(ants[i].tour)
                    self._new_ant.cost = self._cost_function(self._new_ant.tour)

                    if self._new_ant.cost < ants[i].cost:

                        ants[i] = copy.deepcopy(self._new_ant)

                        if self._new_ant.cost < self._best_ant.cost:

                            self._best_ant = copy.deepcopy(self._new_ant)



        return ants

    def _apply_mutation(self, tour):

        def swap(path):

            indices = [int(i) for i in np.random.choice(range(len(path)), 2, replace=False)]
            min_index, max_index = min(indices), max(indices)

            new_path = path.copy()
            new_path[min_index], new_path[max_index] = path[max_index], path[min_index]

            return new_path

        def insertion(path):
            indices = [int(i) for i in np.random.choice(range(len(path)), 2, replace=False)]
            min_index, max_index = min(indices), max(indices)

            method_index = self._roulette_wheel_selection(np.random.dirichlet([0.5, 0.5]))

            if method_index == 0:

                new_path = np.concatenate((path[:min_index + 1],
                                           path[max_index:max_index + 1],
                                           path[min_index + 1: max_index],
                                           path[max_index + 1:]))


            else:

                new_path = np.concatenate((path[:min_index],
                                           path[min_index + 1:max_index + 1],
                                           path[min_index: min_index + 1],
                                           path[max_index + 1:]))

            new_path = [int(i) for i in new_path]
            return new_path

        def reversion(path):
            indices = [int(i) for i in np.random.choice(range(len(path)), 2, replace=False)]
            min_index, max_index = min(indices), max(indices)

            new_path = np.concatenate((path[:min_index],
                                       np.flip(path[min_index: max_index]),
                                       path[max_index:]))
            new_path = [int(i) for i in new_path]

            return new_path

        method = self._roulette_wheel_selection(np.random.dirichlet([0.3, 0.3, 0.4]))
        if method == 0:

            new_tour = swap(tour)

        elif method == 1:

            new_tour = insertion(tour)

        else:

            new_tour = reversion(tour)

        return new_tour




    def _update_pheromone(self):

        for i in range(self._number_of_ants):

            tour = self._ants[i].tour

            for k in range(len(tour) - 1):

                self._pheromone[tour[k], tour[k + 1]] = self._pheromone[tour[k], tour[k + 1]] + (self._pheromone_constant / self._ants[i].cost)
                self._pheromone[tour[k + 1], tour[k]] = self._pheromone[tour[k], tour[k + 1]]

            self._pheromone[tour[-1], tour[0]] = self._pheromone[tour[-1], tour[0]] + (self._pheromone_constant / self._ants[i].cost)
            self._pheromone[tour[0], tour[-1]] = self._pheromone[tour[-1], tour[0]]
        self._pheromone *= (1 - self._pheromone_evaporation_rate)

    def run(self):

        tic = time.time()

        for iter_main in range(self._max_iteration):

            self._ants = self._initialize_ants()

            self._ants = self._evaluation_ants(self._ants)

            self._best_cost.append(self._best_ant.cost)

            self._update_pheromone()


        toc = time.time()

        if self._plot:

            tour = self._best_ant.tour
            location_x = self._model_data["location_x"]
            location_y = self._model_data["location_y"]

            os.makedirs("./figures", exist_ok=True)

            plt.figure(dpi=300, figsize=(16, 10))
            plt.plot(range(self._max_iteration), self._best_cost)
            plt.xlabel("Iteration")
            plt.ylabel("Cost")
            plt.title("Travelling Salesman Problem Using Ant Colony Optimization", fontweight="bold")
            plt.savefig("./figures/cost_function.png")

            plt.figure(dpi=300, figsize=(16, 10))
            plt.scatter(location_x, location_y, marker="o", s=10, facecolors="none", edgecolors="green")
            for i in range(len(tour)):

                plt.text(location_x[tour[i]], location_y[tour[i]], str(tour[i]))

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
