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
                 heuristic_information,
                 pheromone,
                 Q,
                 exponential_weight_for_heuristic,
                 exponential_weight_for_pheromone,
                 pheromone_evaporation_rate,
                 mutation=False,
                 plot=False):

        self._cost_function = cost_function
        self._max_iteration = max_iteration
        self._number_of_ants = number_of_ants
        self._heuristic_information = heuristic_information
        self._pheromone = pheromone
        self._exponential_weight_heuristic = exponential_weight_for_heuristic
        self._exponential_weight_pheromone = exponential_weight_for_pheromone
        self._pheromone_evaporation_rate = pheromone_evaporation_rate
        self._mutate = mutation
        self._plot = plot
        self._best_costs = []
        self._best_ant = self._Ant()
        self._best_ant.cost = 1e20
        self._model_data = Data.load()
        self._Q = Q
        self._mutation_count = 0

    @staticmethod
    def _roulette_wheel_selection(probs):

        random_number = np.random.rand()
        probs_cum_sum = np.cumsum(probs)
        return int(np.argwhere(random_number < probs_cum_sum)[0][0])

    def _initialize_ants(self):

        population = [self._Ant() for _ in range(self._number_of_ants)]

        for i in range(self._number_of_ants):

            population[i].tour = [int(i) for i in np.random.choice(range(len(self._model_data["location_x"])), 1)]

            for j in range(1, len(self._model_data["location_x"])):

                last_visited_node = population[i].tour[-1]

                P = np.power(self._pheromone[last_visited_node, :], self._exponential_weight_pheromone) * \
                np.power(self._heuristic_information[last_visited_node, :], self._exponential_weight_heuristic)

                P[population[i].tour] = 0

                P /= np.sum(P)

                population[i].tour.append(self._roulette_wheel_selection(P))

        return population

    def _evaluate_ants(self, ants):

        for i in range(len(ants)):

            ants[i].cost = self._cost_function(ants[i].tour)


            if self._mutate:

                new_tour = self._mutation(ants[i].tour)

                new_cost = self._cost_function(new_tour)

                if new_cost < ants[i].cost:

                    ants[i].tour = copy.deepcopy(new_tour)
                    ants[i].cost = copy.deepcopy(new_cost)


            if ants[i].cost < self._best_ant.cost:

                self._best_ant = copy.deepcopy(ants[i])

                new_tour_best_ant = self._mutation(self._best_ant.tour)

                new_tour_best_ant_cost = self._cost_function(new_tour_best_ant)

                if new_tour_best_ant_cost < self._best_ant.cost:

                    self._best_ant.tour = copy.deepcopy(new_tour_best_ant)
                    self._best_ant.cost = copy.deepcopy(new_tour_best_ant_cost)

    def _mutation(self, tour):

        tour = copy.deepcopy(np.array(tour))

        indices = [int(i) for i in np.random.choice(range(len(self._model_data["location_x"])), 2, replace=False)]
        min_index, max_index = min(indices), max(indices)

        def insertion(path, first_index, second_index):

            method = self._roulette_wheel_selection(np.random.dirichlet([0.5, 0.5]))

            if method == 0:

                out = np.concatenate((path[:first_index + 1],
                                      path[second_index:second_index + 1],
                                      path[first_index + 1: second_index],
                                      path[second_index + 1:]))

                out = [int(i) for i in out]

            else:

                out = np.concatenate((path[:first_index],
                                      path[first_index + 1: second_index + 1],
                                      path[first_index: first_index + 1],
                                      path[second_index + 1:]))

                out = [int(i) for i in out]

            return out

        def swap(path, first_index, second_index):

            out = np.copy(path)
            out[first_index], out[second_index] = path[second_index], path[first_index]
            out = [int(i) for i in out]
            return out

        def reversion(path, first_index, second_index):

            out = np.concatenate((path[:first_index],
                                  np.flip(path[first_index:second_index]),
                                  path[second_index:]))

            out = [int(i) for i in out]

            return out

        method_selection = self._roulette_wheel_selection(np.random.dirichlet([0.3, 0.3, 0.4]))

        if method_selection == 0:

            return insertion(tour, min_index, max_index)

        elif method_selection == 1:

            return swap(tour, min_index, max_index)

        else:

            return reversion(tour, min_index, max_index)

    def _update_pheromone(self, population):

        for i in range(len(population)):

            tour = population[i].tour

            for k in range(len(tour) - 1):

                self._pheromone[tour[k], tour[k + 1]] += self._Q / population[i].cost

                self._pheromone[tour[k + 1], tour[k]] = self._pheromone[tour[k], tour[k + 1]]

            self._pheromone[tour[-1], tour[0]] += self._Q / population[i].cost
            self._pheromone[tour[0], tour[-1]] = self._pheromone[tour[-1], tour[0]]

        self._pheromone = np.multiply(self._pheromone, 1 - self._pheromone_evaporation_rate)

    def run(self):

        tic = time.time()

        for iter_main in range(self._max_iteration):

            ants = self._initialize_ants()

            self._evaluate_ants(ants)

            self._update_pheromone(ants)

            self._best_costs.append(self._best_ant.cost)



        toc = time.time()

        if self._plot:

            os.makedirs("./figures", exist_ok=True)

            plt.figure(dpi=300, figsize=(16, 10))
            plt.plot(range(self._max_iteration), self._best_costs)
            plt.xlabel("Number of iteration")
            plt.ylabel("Best Cost")
            plt.title("Travelling Salesman Problem Using Ant Colony Optimization", fontweight="bold")
            plt.savefig("./figures/cost_function.png")

            plt.figure(dpi=300, figsize=(16, 10))
            plt.scatter(self._model_data["location_x"],
                        self._model_data["location_y"], marker="o", c="blue", s=8)
            for i in range(len(self._model_data["location_x"])):

                plt.text(self._model_data["location_x"][i] + 0.025,
                         self._model_data["location_y"][i] + 0.025, str(i))

            tour = self._best_ant.tour
            location_x = self._model_data["location_x"]
            location_y = self._model_data["location_y"]

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

