import matplotlib.pyplot as plt
import numpy as np
import copy
import time
import os


class ACOR:

    class _Ant:

        def __init__(self):

            self.position = None
            self.cost = None

    def __init__(self,
                 cost_function,
                 min_range_of_variables,
                 max_range_of_variables,
                 number_of_variables,
                 max_iteration,
                 number_of_ants,
                 number_of_samples,
                 intensification_factor,
                 deviation_factor):

        self._cost_function = cost_function
        self._min_range_of_variables = min_range_of_variables
        self._max_range_of_variables = max_range_of_variables
        self._number_of_variables = number_of_variables
        self._max_iteration = max_iteration
        self._number_of_ants = number_of_ants
        self._number_of_samples = number_of_samples
        self._int_factor = intensification_factor
        self._dev_rate = deviation_factor
        self._ants = None
        self._new_ants = None
        self._best_cost = []
        self._mean = None
        self._std = None
        self._probs = None

    @staticmethod
    def _roulette_wheel_selection(probs):

        random_number = np.random.rand()

        probs_cumsum = np.cumsum(probs)

        return int(np.argwhere(random_number < probs_cumsum)[0][0])

    def _initialize_evaluation_ants(self):

        ants = [self._Ant() for _ in range(self._number_of_ants)]

        for i in range(self._number_of_ants):

            ants[i].position = np.random.uniform(self._min_range_of_variables,
                                                 self._max_range_of_variables,
                                                 (1, self._number_of_variables))

            ants[i].cost = self._cost_function(ants[i].position)

        return ants

    @staticmethod
    def _sort_ants(ants):

        ants_cost_argsort = [int(i) for i in np.argsort([ants[j].cost for j in range(len(ants))])]

        sorted_ants = [ants[i] for i in ants_cost_argsort]

        return sorted_ants

    def _calc_mean(self):

        mean = np.reshape([self._ants[i].position for i in range(self._number_of_ants)],
                          (self._number_of_ants, self._number_of_variables))

        return mean

    def _calc_std(self):

        std = np.zeros_like(self._mean)

        for i in range(self._number_of_ants):

            for j in range(self._number_of_ants):

                std[i, :] += np.abs(self._mean[i, :] - self._mean[j, :])

        std /= (self._number_of_ants - 1)
        std *= self._dev_rate

        return std

    def _calc_prob(self):

        weights = (1 / (np.sqrt(2 * np.pi) * self._int_factor * self._number_of_ants)) * \
            np.exp((-1 / 2) * np.square(np.arange(0, self._number_of_ants) / self._int_factor * self._number_of_ants))

        weights /= np.sum(weights)

        return weights

    def _select_samples(self):

        samples = [self._Ant() for _ in range(self._number_of_samples)]

        for i in range(self._number_of_samples):

            samples[i].position = np.zeros((1, self._number_of_variables))

            for k in range(self._number_of_variables):

                kernel = self._roulette_wheel_selection(self._probs)

                samples[i].position[0, k] = self._mean[kernel, k] + self._std[kernel, k] * np.random.randn()

            samples[i].cost = self._cost_function(samples[i].position)

        return samples

    def run(self):

        tic = time.time()

        self._ants = self._initialize_evaluation_ants()

        self._ants = self._sort_ants(self._ants)

        self._probs = self._calc_prob()

        for iter_main in range(self._max_iteration):

            self._mean = self._calc_mean()

            self._std = self._calc_std()

            self._new_ants = self._select_samples()

            self._ants.extend(self._new_ants)

            self._ants = self._sort_ants(self._ants)

            self._ants = self._ants[:self._number_of_ants]

            self._best_cost.append(self._ants[0].cost)

        toc = time.time()

        os.makedirs("./figures", exist_ok=True)

        plt.figure(dpi=300, figsize=(10, 6))
        plt.semilogy(range(self._max_iteration), self._best_cost)
        plt.xlabel("Number of Iteration")
        plt.ylabel("Cost")
        plt.title(r"Sphere Function using Ant Colony $Optimization_{R}$")
        plt.savefig("./figures/cost_function.png")

        return self._ants[0], toc - tic
