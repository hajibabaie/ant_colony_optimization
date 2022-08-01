from ant_colony_optimization.knapsack_problem_binary.data import Data
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
            self.solution_parsed = None

    def __init__(self,
                 cost_function,
                 max_iteration,
                 number_of_ants,
                 pheromone,
                 pheromone_constant_rate,
                 pheromone_exponent_rate,
                 pheromone_evaporation_rate,
                 heuristic_information,
                 heuristic_exponent_rate,
                 plot_cost=False):

        self._cost_function = cost_function
        self._max_iteration = max_iteration
        self._number_of_ants = number_of_ants
        self._ants = None
        self._pheromone = pheromone
        self._pheromone_constant_rate = pheromone_constant_rate
        self._pheromone_exponent_rate = pheromone_exponent_rate
        self._pheromone_evaporation_rate = pheromone_evaporation_rate
        self._heuristic_information = heuristic_information
        self._heuristic_exponent_rate = heuristic_exponent_rate
        self._best_ant = self._Ant()
        self._best_ant.cost = 1e20
        self._best_cost = []
        self._plot_cost = plot_cost
        self._model_data = Data.load()

    @staticmethod
    def _roulette_wheel_selection(probs):

        random_number = np.random.rand()
        probs_cumsum = np.cumsum(probs)

        return int(np.argwhere(random_number < probs_cumsum)[0][0])

    def _initialize_ants(self):

        ants = [self._Ant() for _ in range(self._number_of_ants)]

        for i in range(self._number_of_ants):

            for k in range(len(self._model_data["values"])):

                P = np.power(self._pheromone[:, k], self._pheromone_exponent_rate) * \
                    np.power(self._heuristic_information[:, k], self._pheromone_exponent_rate)

                P /= np.sum(P)

                ants[i].position.append(self._roulette_wheel_selection(P))

        return ants

    def _evaluate_ants(self, ants):

        for i in range(len(ants)):

            ants[i].solution_parsed, \
            ants[i].cost = self._cost_function(ants[i].position)

            if ants[i].cost < self._best_ant.cost:

                self._best_ant = copy.deepcopy(ants[i])

        return ants

    def _update_pheromone(self):

        for i in range(self._number_of_ants):

            for k in range(len(self._ants[i].position)):

                self._pheromone[self._ants[i].position[k], k] += self._pheromone_constant_rate / self._ants[i].cost

        self._pheromone *= (1 - self._pheromone_evaporation_rate)

    def run(self):

        tic = time.time()

        for iter_main in range(self._max_iteration):

            self._ants = self._initialize_ants()

            self._ants = self._evaluate_ants(self._ants)

            self._update_pheromone()

            self._best_cost.append(self._best_ant.cost)



        toc = time.time()

        if self._plot_cost:

            os.makedirs("./figures", exist_ok=True)
            plt.figure(dpi=300, figsize=(10, 6))
            plt.plot(range(self._max_iteration), self._best_cost)
            plt.xlabel("Number of Iteration")
            plt.ylabel("Cost")
            plt.title("Binary Knapsack Problem Using Ant Colony Optimization", fontweight="bold")
            plt.savefig("./figures/cost_function.png")

        return self._best_ant, toc - tic
