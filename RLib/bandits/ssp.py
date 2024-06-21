import random
import numpy as np
from math import log, sqrt
import networkx as nx


def find_all_paths(graph, start_node, end_node):
    return list(nx.all_simple_paths(graph, start_node, end_node))


def calculate_path_weights(graph, paths, weight='length'):
    path_weights = []
    for path in paths:
        total_weight = 0
        for i in range(len(path) - 1):
            # node, next_node = path[i], path[i + 1]
            total_weight += graph.edges[path[i], path[i + 1]][weight]
        path_weights.append(total_weight)
    return path_weights


def train_bandit(bandit, num_rounds, distribution='normal'):
    regrets = []
    for t in range(1, num_rounds + 1):
        chosen_arm = bandit.select_arm()
        # Recompensa negativa ya que buscamos minimizar el costo
        reward = min(- random_time(path_lengths[chosen_arm],
                                   25, distribution), 0)
        bandit.pull(chosen_arm, reward)


class MultiArmedBandit:
    def __init__(self, paths_mean_cost):
        self.num_paths = len(paths_mean_cost)
        self.mean_rewards = paths_mean_cost
        self.optimal_path = max(paths_mean_cost)
        self.rewards = []
        self.regret_history = []
        self.average_regret_history = []
        self.pseudo_regret_history = []
        self.arm_pulls = [0] * self.num_paths
        self.arm_rewards = [0] * self.num_paths
        self.arms_history = []
        self.strategy = 'Generic Multi-Armed Bandit'

    def K(self):
        return self.num_paths

    def regret(self, t):
        """Returns the random regret incurred so far.

        Args:
            t (int): Number of training rounds.

        Returns:
            float: Random regret incurred so far. Reference: Tor Lattimore and Csaba Szepesvári. Bandit Algorithms. Cambridge University Press, 2020. Page 55.
        """
        return t * self.optimal_path - sum(self.rewards[:t])

    def average_regret(self, t):
        """Returns the average regret incurred so far.

        Args:
            t (int): Number of training rounds.

        Returns:
            float: Average regret incurred so far. Reference: Tor Lattimore and Csaba Szepesvári. Bandit Algorithms. Cambridge University Press, 2020. Page 55.
        """

        return self.regret(t) / t

    def pseudo_regret(self, t):
        return t * self.optimal_path - sum(np.array(self.mean_rewards) * np.array(self.arm_pulls))

    def update_regret_history(self, t):
        self.regret_history.append(self.regret(t))
        self.average_regret_history.append(self.average_regret(t))
        self.pseudo_regret_history.append(self.pseudo_regret(t))

    def select_arm(self):
        raise NotImplementedError(
            "Subclasses should implement select_arm method.")

    def pull(self, chosen_arm, reward):
        self.arm_pulls[chosen_arm] += 1
        self.arms_history.append(chosen_arm)
        n = self.arm_pulls[chosen_arm]
        current_value = self.arm_rewards[chosen_arm]
        new_value = current_value + (reward - current_value) / n
        self.arm_rewards[chosen_arm] = new_value
        self.rewards.append(reward)
        self.update_regret_history(len(self.rewards))


class EGreedyMultiArmedBandit(MultiArmedBandit):
    def __init__(self, paths_mean_cost):
        super().__init__(paths_mean_cost)
        self.strategy = 'Epsilon-Greedy'

    def select_arm(self):
        epsilon = 0.1
        if random.random() > epsilon:
            return self.arm_rewards.index(max(self.arm_rewards))
        else:
            return random.randint(0, self.num_paths - 1)


class UCBMultiArmedBandit(MultiArmedBandit):
    def __init__(self, paths_mean_cost, c=2):
        super().__init__(paths_mean_cost)
        self.strategy = 'UCB'
        self.c = c

    def select_arm(self):
        for i in range(self.num_paths):
            if self.arm_pulls[i] == 0:
                return i
        ucb_values = [
            self.arm_rewards[i] +
            sqrt(self.c * log(len(self.rewards)) / self.arm_pulls[i])
            for i in range(self.num_paths)
        ]
        return ucb_values.index(max(ucb_values))


class EXP3MultiArmedBandit(MultiArmedBandit):
    def __init__(self, paths_mean_cost, eta=0.1):
        super().__init__(paths_mean_cost)
        self.strategy = 'EXP3'
        self.S_hat = np.zeros(self.num_paths)
        self.total_pulls = 0
        self.eta = str(eta)

    def calculate_eta(self, t):
        n = self.K()
        eta = eval(self.eta)
        return eta

    def calculate_probabilities(self):
        eta = self.calculate_eta(self.total_pulls+1)
        max_S_hat = np.max(self.S_hat)
        # Valores normalizados
        exp_values = np.exp(eta * (self.S_hat - max_S_hat))
        total_exp_values = np.sum(exp_values)
        probabilities = exp_values / total_exp_values
        return probabilities

    def select_arm(self):
        probabilities = self.calculate_probabilities()
        return np.random.choice(self.num_paths, p=probabilities)

    def pull(self, chosen_arm, reward):
        self.total_pulls += 1
        self.arm_pulls[chosen_arm] += 1
        self.rewards.append(reward)

        probabilities = self.calculate_probabilities()

        # Actualizar S_hat para cada brazo
        for i in range(self.num_paths):
            if i == chosen_arm:
                self.S_hat[i] += 1 - (1 - reward) / probabilities[i]
            else:
                self.S_hat[i] += 1
        self.update_regret_history(len(self.rewards))


class BoltzmannBandit(MultiArmedBandit):
    def __init__(self, paths_mean_cost, eta=0.1):
        super().__init__(paths_mean_cost)
        self.strategy = 'Boltzmann'
        self.arm_rewards = np.zeros(self.num_paths)
        self.total_pulls = 0
        self.eta = str(eta)

    def calculate_eta(self, t):
        n = self.K()
        eta = eval(self.eta)
        return eta

    def calculate_probabilities(self):
        eta = self.calculate_eta(self.total_pulls+1)
        max_rewards = np.max(self.arm_rewards)
        # Valores normalizados
        exp_values = np.exp(eta * (self.arm_rewards - max_rewards))
        total_exp_values = np.sum(exp_values)
        probabilities = exp_values / total_exp_values
        return probabilities

    def select_arm(self):
        probabilities = self.calculate_probabilities()
        return np.random.choice(self.num_paths, p=probabilities)


if __name__ == '__main__':
    import sys
    import os

    # Añadir el directorio superior a RLib al PYTHONPATH
    sys.path.append(os.path.abspath(
        os.path.join(os.path.dirname(__file__), '../..')))
    # print("PYTHONPATH:", sys.path)

    from RLib.graphs.perceptron import create_perceptron_graph, plot_network_graph
    from RLib.distributions.distributions import expected_time, random_time
    from RLib.bandits.utils.plot import plot_bandits_regret

    # Create a perceptron graph
    nodes_by_layer = [1, 20, 1]
    graph = create_perceptron_graph(nodes_by_layer, 100, 2000)
    # Encontrar todos los caminos desde 'Entrada' hasta 'Salida'
    start_node = ('Entrada', 0)
    end_node = ('Salida', 0)
    # Obtener los caminos posibles
    all_paths = find_all_paths(graph, start_node, end_node)
    # Calcular el largo y el costo medio de cada camino
    path_lengths = calculate_path_weights(graph, all_paths)
    cost_distribution = 'normal'
    path_costs = list(
        map(lambda x: - expected_time(x, 25, cost_distribution), path_lengths))

    bandits_list = []
    eta = "log(t+1)"
    for i in range(5):
        # Crear el Multi-Armed Bandit
        bandit_eps = EGreedyMultiArmedBandit(path_costs)
        # Crear el UCB Multi-Armed Bandit
        bandit_ucb = UCBMultiArmedBandit(path_costs)
        # Crear el EXP3 Multi-Armed Bandit
        bandit_exp3 = EXP3MultiArmedBandit(path_costs, eta=eta)
        # Crear el Boltzmann Multi-Armed Bandit
        bandit_boltz = BoltzmannBandit(path_costs, eta=eta)

        # Realizar entrenamientos
        train_bandit(bandit_eps, 10000, cost_distribution)
        train_bandit(bandit_ucb, 10000, cost_distribution)
        train_bandit(bandit_exp3, 10000, cost_distribution)
        train_bandit(bandit_boltz, 10000, cost_distribution)

        bandits_list.append(bandit_eps)
        bandits_list.append(bandit_ucb)
        bandits_list.append(bandit_exp3)
        bandits_list.append(bandit_boltz)

    fig = plot_bandits_regret(
        bandits_list, criteria="regret")
    fig.show()
    fig = plot_bandits_regret(
        bandits_list, criteria="average regret")
    fig.show()
    fig = plot_bandits_regret(
        bandits_list, criteria="pseudo regret")
    fig.show()
    fig = plot_bandits_regret(
        bandits_list, criteria="rewards")
    fig.show()
    fig = plot_bandits_regret(
        bandits_list, criteria="pulls")
    fig.show()
