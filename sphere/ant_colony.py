import matplotlib.pyplot as plt
import numpy as np
import time
import copy
import os


class ACOR:

    class _Ant:

        def __init__(self):

            self.position = {}
            self.cost = None

    def __init__(self,
                 cost_function,
                 type_number_of_variables,
                 min_range_of_variables,
                 max_range_of_variables,
                 max_iteration,
                 number_of_ants,
                 number_of_new_samples,
                 intensification_factor,
                 deviation_rate,
                 plot_solution=False):

        self._cost_function = cost_function
        self._type_number_of_variables = type_number_of_variables
        self._min_range_of_variables = min_range_of_variables
        self._max_range_of_variables = max_range_of_variables
        self._max_iteration = max_iteration
        self._number_of_ants = number_of_ants
        self._ants = None
        self._number_of_new_samples = number_of_new_samples
        self._new_ants = None
        self._intensification_factor = intensification_factor
        self._deviation_rate = deviation_rate
        self._plot_solution = plot_solution
        self._best_ant = None
        self._best_cost = []
        self._probs_ants = None
        self._mean = None
        self._std = None


    @staticmethod
    def _roulette_wheel_selection(probs):

        random_number = np.random.random()

        probs_cumsum = np.cumsum(probs)

        return int(np.argwhere(random_number <= probs_cumsum)[0][0])

    def _initialize_ants(self):

        ants = [self._Ant() for _ in range(self._number_of_ants)]

        def initialize_ants_real1D(population):

            for i in range(len(population)):

                population[i].position["real1D"] = {}

                for j in range(len(self._type_number_of_variables["real1D"])):

                    population[i].position["real1D"][j] = \
                        np.random.uniform(self._min_range_of_variables,
                                          self._max_range_of_variables,
                                          (1, self._type_number_of_variables["real1D"][j]))

            return population

        for type_of_variable in self._type_number_of_variables.keys():

            if type_of_variable == "real1D":

                ants = initialize_ants_real1D(ants)

        return ants

    def _evaluate_cost(self, population):

        for i in range(len(population)):

            population[i].cost = self._cost_function(population[i].position)

        return population

    @staticmethod
    def _sort(population):

        population_argsort = np.argsort([population[i].cost for i in range(len(population))])

        population = [population[int(i)] for i in population_argsort]

        return population

    def _calculate_probs(self, population):

        k = len(population)

        weights = (1 / (np.sqrt(2 * np.pi) * self._intensification_factor * k)) * \
            np.exp((-1/2) * np.square((np.arange(1, k + 1) - 1)) / np.square(self._intensification_factor * k))

        weights /= np.sum(weights)

        return weights

    @staticmethod
    def _calculate_mean(population):

        return np.reshape([population[i].position["real1D"][0]
                           for i in range(len(population))], (len(population),
                                                              int(population[0].position["real1D"][0].shape[1])))

    def _calculate_std(self, mean):

        std = np.zeros_like(mean)
        for i in range(int(std.shape[0])):
            for j in range(int(std.shape[0])):

                std[i, :] += np.abs(mean[i, :] - mean[j, :])

        std /= (int(std.shape[0]) - 1)
        std *= self._deviation_rate

        return std

    def _new_population(self, mean, std, probs):

        new_ants = [self._Ant() for _ in range(self._number_of_new_samples)]

        for i in range(self._number_of_new_samples):

            new_ants[i].position["real1D"] = {}
            new_ants[i].position["real1D"][0] = np.zeros((1, self._type_number_of_variables["real1D"][0]))

            for k in range(int(mean.shape[1])):

                kernel = self._roulette_wheel_selection(probs)

                new_ants[i].position["real1D"][0][0, k] = mean[kernel, k] + std[kernel, k] * np.random.randn()

        return new_ants

    def run(self):

        tic = time.time()

        self._ants = self._initialize_ants()

        self._ants = self._evaluate_cost(self._ants)

        self._ants = self._sort(self._ants)

        self._best_ant = copy.deepcopy(self._ants[0])

        self._probs_ants = self._calculate_probs(self._ants)

        for iter_main in range(self._max_iteration):

            self._mean = self._calculate_mean(self._ants)

            self._std = self._calculate_std(self._mean)

            self._new_ants = self._new_population(self._mean, self._std, self._probs_ants)

            self._new_ants = self._evaluate_cost(self._new_ants)

            self._ants.extend(self._new_ants)

            self._ants = self._sort(self._ants)

            self._ants = self._ants[:self._number_of_ants]

            self._best_ant = self._ants[0]

            self._best_cost.append(self._best_ant.cost)

        toc = time.time()

        if self._plot_solution:

            os.makedirs("./figures", exist_ok=True)

            plt.figure(dpi=300, figsize=(10, 6))
            plt.semilogy(range(self._max_iteration), self._best_cost)
            plt.title(r"Sphere Function Using Ant Colony $Optimization_{R}$")
            plt.xlabel("Number of Iteration")
            plt.ylabel("Cost")
            plt.savefig("./figures/cost_function.png")

        return self._best_ant, toc - tic
