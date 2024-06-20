import random
import numpy as np
from math import log, sqrt
import networkx as nx

def find_all_paths(graph, start_node, end_node):
    return list(nx.all_simple_paths(graph, start_node, end_node))

def calculate_path_weights(graph, paths, weight='length'):
    path_costs = []
    for path in paths:
        total_cost = 0
        for i in range(len(path) - 1):
            # node, next_node = path[i], path[i + 1]
            total_cost += graph.edges[path[i], path[i + 1]][weight]
        path_costs.append(total_cost)
    return path_costs

def train_bandit(bandit, num_rounds):
    regrets = []
    for t in range(1, num_rounds + 1):
        chosen_arm = bandit.select_arm()
        # Simular la recompensa de la acción
        # reward = - path_costs[chosen_arm]  # Recompensa negativa ya que buscamos minimizar el costo
        reward =  + random_time(path_lengths[chosen_arm], 25)  # Recompensa negativa ya que buscamos minimizar el costo
        bandit.pull(chosen_arm, reward)
        regrets.append(max(path_costs) - sum(bandit.rewards) / t )
    bandit.regret_history = regrets

class EGreedyMultiArmedBandit:
    def __init__(self, paths_mean_cost):
        self.num_paths = len(paths_mean_cost)
        self.paths_costs = paths_mean_cost
        self.optimal_path = max(paths_mean_cost)
        self.rewards = []
        self.arm_pulls = [0] * self.num_paths
        self.arm_rewards = [0] * self.num_paths
        self.strategy = 'e-greedy'

    def K(self):
        return self.num_paths

    def regret(self):
        return self.optimal_path - sum(self.rewards)

    def select_arm(self):
        # Implementación simple usando epsilon-greedy
        epsilon = 0.1
        if random.random() > epsilon:
            # Exploit: seleccionar el brazo con la mejor recompensa promedio
            return self.arm_rewards.index(max(self.arm_rewards))
        else:
            # Explore: seleccionar un brazo aleatoriamente
            return random.randint(0, self.num_paths - 1)

    def pull(self, chosen_arm, reward):
        # Actualizar la recompensa y el número de selecciones del brazo elegido
        self.arm_pulls[chosen_arm] += 1
        n = self.arm_pulls[chosen_arm]
        current_value = self.arm_rewards[chosen_arm]
        new_value = current_value + (reward - current_value) / n
        self.arm_rewards[chosen_arm] = new_value
        self.rewards.append(reward)


class UCBMultiArmedBandit:
    def __init__(self, paths_mean_cost):
        self.num_paths = len(paths_mean_cost)
        self.paths_costs = paths_mean_cost
        self.optimal_path = max(paths_mean_cost)
        self.rewards = []
        self.regrets = []
        self.arm_pulls = [0] * self.num_paths
        self.arm_rewards = [0] * self.num_paths
        self.total_pulls = 0
        self.strategy = 'UCB'

    def K(self):
        return self.num_paths

    def regret(self):
        n = len(self.rewards)
        return n * self.optimal_path - sum(self.rewards)

    def select_arm(self):
        for i in range(self.num_paths):
            if self.arm_pulls[i] == 0:
                return i
        ucb_values = [
            self.arm_rewards[i] +
            sqrt(2 * log(self.total_pulls) / self.arm_pulls[i])
            for i in range(self.num_paths)
        ]
        return ucb_values.index(max(ucb_values))

    def pull(self, chosen_arm, reward):
        self.total_pulls += 1
        self.arm_pulls[chosen_arm] += 1
        n = self.arm_pulls[chosen_arm]
        current_value = self.arm_rewards[chosen_arm]
        new_value = current_value + (reward - current_value) / n
        self.arm_rewards[chosen_arm] = new_value
        self.rewards.append(reward)


class EXP3MultiArmedBandit:
    def __init__(self, paths_mean_cost, eta=0.1):
        """ 
            - eta = 0.1 (o cualquier valor de eta constante de tipo float)
            - eta = 't'
            - eta = 'log(t)'
            - eta = 'sqrt(t)'
        """
        self.num_paths = len(paths_mean_cost)
        self.paths_costs = paths_mean_cost
        self.optimal_path = max(paths_mean_cost)
        self.rewards = []
        self.arm_pulls = [0] * self.num_paths
        self.S_hat = np.zeros(self.num_paths)  # Inicializa S_hat en 0
        self.total_pulls = 0
        self.eta = str(eta)
        self.strategy = 'EXP3'

    def K(self):
        return self.num_paths

    def regret(self):
        return self.optimal_path - sum(self.rewards)

    def calculate_eta(self, t):
        n = self.K()
        eta = eval(self.eta)
        return eta

    def calculate_probabilities(self):
        eta = self.calculate_eta(self.total_pulls+1)
        max_S_hat = np.max(self.S_hat)
        exp_values = np.exp(eta * (self.S_hat - max_S_hat)
                            )  # Valores normalizados
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


class BoltzmannBandit:
    def __init__(self, paths_mean_cost, eta=0.1):
        """ 
            - eta = 0.1 (o cualquier valor de eta constante de tipo float)
            - eta = 't'
            - eta = 'log(t)'
            - eta = 'sqrt(t)'
        """
        self.num_paths = len(paths_mean_cost)
        self.paths_costs = paths_mean_cost
        self.optimal_path = max(paths_mean_cost)
        self.rewards = []
        self.arm_pulls = [0] * self.num_paths
        # Inicializa las recompensas acumuladas
        self.arm_rewards = np.zeros(self.num_paths)
        self.total_pulls = 0
        self.eta = str(eta)
        self.strategy = 'Boltzmann'

    def K(self):
        return self.num_paths

    def regret(self):
        return self.optimal_path - sum(self.rewards)

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

    def pull(self, chosen_arm, reward):
        self.total_pulls += 1
        self.arm_pulls[chosen_arm] += 1
        self.rewards.append(reward)
        self.arm_rewards[chosen_arm] += reward


if __name__ == '__main__':
    import sys
    import os

    # Añadir el directorio superior a RLib al PYTHONPATH
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    # print("PYTHONPATH:", sys.path)

    from RLib.graphs.perceptron import create_perceptron_graph, plot_network_graph
    from RLib.distributions.distributions import expected_time, random_time
    from RLib.bandits.utils.plot import plot_bandits_regret

    # Create a perceptron graph
    nodes_by_layer = [1, 40, 1]
    graph = create_perceptron_graph(nodes_by_layer, 100, 2000)
    # Encontrar todos los caminos desde 'Entrada' hasta 'Salida'
    start_node = ('Entrada', 0)
    end_node = ('Salida', 0)
    # Obtener los caminos posibles
    all_paths = find_all_paths(graph, start_node, end_node)
    # Calcular el largo y el costo medio de cada camino
    path_lengths = calculate_path_weights(graph, all_paths)
    path_costs = list(map(lambda x: expected_time(x, 25), path_lengths))
    
    # Crear el Multi-Armed Bandit
    bandit_eps = EGreedyMultiArmedBandit(path_costs)
    # Crear el UCB Multi-Armed Bandit
    bandit_ucb = UCBMultiArmedBandit(path_costs)
    # Crear el EXP3 Multi-Armed Bandit
    eta = "log(t+1)"
    bandit_exp3 = EXP3MultiArmedBandit(path_costs, eta=eta)
    # Crear el Boltzmann Multi-Armed Bandit
    bandit_boltz = BoltzmannBandit(path_costs, eta=eta)

    # Realizar entrenamientos
    train_bandit(bandit_eps, 10000)
    train_bandit(bandit_ucb, 10000)
    train_bandit(bandit_exp3, 10000)
    train_bandit(bandit_boltz, 10000)

    fig = plot_bandits_regret([bandit_eps, bandit_ucb, bandit_exp3, bandit_boltz], criteria="regret")
    fig.show()
    fig = plot_bandits_regret([bandit_eps, bandit_ucb, bandit_exp3, bandit_boltz], criteria="cumulative regret")
    fig.show()
    fig = plot_bandits_regret([bandit_eps, bandit_ucb, bandit_exp3, bandit_boltz], criteria="rewards")
    fig.show()
    fig = plot_bandits_regret([bandit_eps, bandit_ucb, bandit_exp3, bandit_boltz], criteria="pulls")
    fig.show()