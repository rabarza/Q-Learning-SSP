from typing import Tuple, Dict, Any
import random
import numpy as np
import networkx as nx
import osmnx as ox
import copy
import sys
import os
# Importar RLib desde el directorio superior
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))  # noqa: E402
from RLib.cost_distributions import (
    expected_time,
    random_time,
)


from RLib.utils.tables import dict_states_actions_zeros, dict_states_zeros, dict_states_actions_constant

##############################################################################################################
# Stochastic Shortest Path
##############################################################################################################


def get_edge_attribute(G: nx.Graph, source: Any, target: Any, weight: str = "length"):
    edge_data = G.get_edge_data(source, target)
    if weight in edge_data:
        edge_length = edge_data[weight]
    else:
        edge_features = next(iter(edge_data.values()))
        assert (
            weight in edge_features
        ), f"No hay atributo {weight} en los arcos entre los nodos {source} y {target}."
        edge_length = edge_features[weight]
    return edge_length


def get_edge_length(G: nx.Graph, source: Any, target: Any) -> float:
    assert G.has_edge(
        source, target), f"No hay arco entre los nodos {source} y {target}."
    return get_edge_attribute(G, source, target, "length")


def get_edge_speed(G: nx.Graph, source, target, avg_speed=25) -> float:
    # Buscar todos los bordes entre source y target
    edge_data = G.get_edge_data(source, target)
    if "speed_kph" in edge_data:
        return edge_data["speed_kph"]
    if isinstance(edge_data, dict) and 0 in edge_data:
        return edge_data[0]['speed_kph'] if 'speed_kph' in edge_data[0] else avg_speed
    else:
        return avg_speed


def get_edge_cost(
    G,
    source: int,
    target: int,
    distribution: str = "expectation-lognormal",
    avg_speed: float = 25,
) -> float:
    """Obtener el costo de un arco entre dos nodos

    Parámetros:
    ----------
        G (nx.Digraph): Grafo con el que se va a trabajar
        source (Any): Nodo source (origen)
        target (Any): Nodo target (destino)
        distribution (str): Distribución de probabilidad que se va a utilizar para obtener el costo estocástico. Si se desea utilizar la esperanza de la distribución, se debe pasar el string "expectation-`distribution`", con un guión separando la palabra "expectation" y el nombre de la distribución. Si se desea obtener una muestra de la distribución, se debe pasar el nombre de la distribución.

    Retorno:
    --------
        stochastic_cost (float): Costo del arco entre los nodos source y target
    """
    edge_length = get_edge_length(G, source, target)
    edge_speed = get_edge_speed(G, source, target, avg_speed)

    # stochastic cost sample from lognormal distribution
    if "expectation" in distribution:
        # get the expected speed and calculate the expected time given the edge length
        time = expected_time(edge_length, edge_speed,
                             distribution.split("-")[1])

    else:
        # generate a sample of the speed and calculate the time given the edge length
        time = random_time(edge_length, edge_speed, distribution)

    stochastic_cost = time
    return stochastic_cost


def get_cumulative_edges_cost(graph: nx.Graph, policy: dict, source: Any, target: Any, distribution: str = "expectation-lognormal") -> float:
    """
    Retorna el costo acumulado de comenzar en el nodo 'source' y llegar al nodo 'target' siguiendo la política 'policy'

    Parámetros:
    -----------
        graph (nx.Graph): Grafo con el que se va a trabajar
        policy (dict): Política de decisión que se va a seguir
        source (int): Nodo de inicio
        target (int): Nodo de destino

    Retorno:
    --------
        cost (float): Costo acumulado de seguir la política 'policy' desde el nodo 'source' al nodo 'target'
    """
    # Obtener el camino más corto desde el nodo de inicio al nodo de destino dada la política
    path = [source]
    while source != target:
        source = policy[source]
        path.append(source)
    # Calcular el costo acumulado del camino más corto
    cost = 0
    for index in range(len(path) - 1):
        cost += get_edge_cost(graph, path[index],
                              path[index + 1], distribution=distribution)
    return cost


class SSPEnv:
    """
    Entorno de aprendizaje para el problema de encontrar el camino más corto en un grafo
    """

    def __init__(self, graph: nx.Graph, start_state: Any, terminal_state: Any, costs_distribution: str = "lognormal", shortest_path: list = None):
        """Constructor de la clase SSPEnv. Inicializa el entorno de aprendizaje con el grafo, el estado inicial y el estado terminal. Notar que al nodo terminal se le agrega un arco recurrente con longitud 0.

        Parámetros:
        ----------
        graph (nx.Graph): Grafo con el que se va a trabajar. 
        start_state (int): Estado inicial
        terminal_state (int): Estado terminal
        """

        assert graph is not None, "El graph no puede ser None"
        self.graph = copy.deepcopy(graph)
        # Añadir un arco recurrente de largo 0 en el nodo terminal
        if not self.graph.has_edge(terminal_state, terminal_state):
            self.graph.add_edge(terminal_state, terminal_state, length=0)
        else:
            self.graph[terminal_state][terminal_state][0]["length"] = 0
        self.num_nodos = graph.number_of_nodes()
        assert start_state is not None, "El estado inicial no puede ser None"
        self.start_state = start_state
        self.current_state = self.start_state
        assert terminal_state is not None, "El estado terminal no puede ser None"
        self.terminal_state = terminal_state
        self.num_states = graph.number_of_nodes()
        self.num_actions = {
            k: v
            for k, v in map(
                lambda item: (item[0], len(item[1])
                              ), nx.to_dict_of_lists(graph).items()
            )
        }  # Diccionario con el número de acciones por estado
        self.adjacency_dict_of_lists = nx.to_dict_of_lists(self.graph)
        self.costs_distribution = costs_distribution
        self.shortest_path = shortest_path

    def __str__(self):
        return f"EnvShortestPath {self.graph}"

    def reset(self):
        """Reiniciar el entorno de aprendizaje"""
        self.current_state = self.start_state
        # print(
        #     f"Entorno reiniciado. Grafo con {self.num_nodos} nodos. Estado inicial: {self.start_state}. Estado final: {self.terminal_state}"
        # )

    def action_set(self, state) -> list:
        """Obtener el conjunto de acciones posibles en un estado

        Parámetros:
        ----------
            state (int): Estado actual

        Retorno:
        --------
            actions (list): Lista de acciones posibles en el estado actual
        """
        return list(self.graph[state])

    def check_state(self, state) -> bool:
        """Verificar si un estado es válido y/o terminal.

        Parámetros:
        ----------
            state (int): Estado a verificar

        Retorno:
        --------
            valid (bool): Indica si el estado es terminal y arroja un error si no es válido
        """
        assert (
            state in self.adjacency_dict_of_lists.keys()
        ), f"El estado {state} no está en el graph"
        return state == self.terminal_state

    def take_action(self, state, action) -> Tuple[int, float, bool, str]:
        """Tomar una acción en el entorno de aprendizaje

        Parámetros:
        ----------
            state (int): Estado actual
            action (int): Acción a tomar

        Retorna:
        --------
            next_state (int): Siguiente estado
            reward (float): Recompensa (o costo) por tomar la acción
            terminated (bool): Indica si el episodio terminó
        """
        # Obtener los arcos del nodo actual (estado actual)
        assert self.graph.has_edge(
            state, action
        ), f"La acción {action} no está en los arcos del nodo {state}"
        # Obtener el siguiente estado, el costo de la acción y verificar si el estado es terminal
        next_state = action
        cost = get_edge_cost(self.graph, state, next_state,
                             self.costs_distribution)
        terminated = self.check_state(state)
        # La recompensa es negativa porque se busca maximizar la recompensa que representa minimizar el costo
        reward = - cost
        self.current_state = next_state
        info = str({"estado": state,
                    "recompensa": reward,
                    "terminado": terminated})
        return next_state, reward, terminated, info


class HardSSPEnv(SSPEnv):
    def __init__(self, graph: nx.Graph, start_state: Any, terminal_state: Any, costs_distribution: str = "lognormal", shortest_path: list = None):
        super().__init__(graph, start_state, terminal_state, costs_distribution, shortest_path)
        if shortest_path is None:
            self.shortest_path = nx.shortest_path(
                self.graph, self.start_state, self.terminal_state)
        self.reset()

    def reset(self):
        self.current_state = self.start_state
        self.current_graph = copy.deepcopy(self.graph)
        self.shortest_path_edges = self.get_path_edges(self.shortest_path)

    def action_set(self, state) -> list:
        return list(self.current_graph[state])

    def remove_edges_to_shortest_path(self):
        """Remover los arcos que comienzan en un nodo que no está en el camino más corto y terminan en un nodo que sí está en el camino más corto
        """
        edges_to_remove = []
        for node in self.current_graph.nodes:
            if node in self.shortest_path:
                continue
            # No remover los arcos del nodo inicial y final
            for neighbor in self.shortest_path[1:-1]:
                if self.current_graph.has_edge(node, neighbor):
                    edges_to_remove.append((node, neighbor))
                    break
        # Remover los arcos
        self.current_graph.remove_edges_from(edges_to_remove)

    def ensure_largest_component(self):
        self.current_graph = ox.truncate.largest_component(
            self.current_graph, strongly=True)

    def check_state(self, state) -> bool:
        if self.terminal_state == state:
            return True
        is_terminal_state = super().check_state(state)
        is_connected = nx.has_path(
            self.current_graph, state, self.terminal_state)
        # print(f"Estado: {state}, Conectado: {is_connected}")
        return is_terminal_state or not is_connected

    def get_path_edges(self, path):
        return [(path[i], path[i + 1]) for i in range(len(path) - 1)]

    def take_action(self, state, action) -> Tuple[int, float, bool, str]:
        # assert self.current_graph.has_edge(
        #     state, action
        # ), f"La acción {action} no está en los arcos del nodo {state}"

        cost = get_edge_cost(self.current_graph, state, action,
                             self.costs_distribution)
        next_state = action if self.current_graph.has_edge(
            state, action) else state

        terminated = self.check_state(state)
        reward = - cost
        self.current_state = next_state
        info = str({"estado": state,
                    "recompensa": reward,
                    "terminado": terminated})
        return next_state, reward, terminated, info


if __name__ == "__main__":
    from RLib.graphs.perceptron import create_perceptron_graph
    G = create_perceptron_graph([1, 10, 10, 1])
    origin_node = ('Entrada', 0)
    target_node = ('Salida', 0)

    env = HardSSPEnv(G, start_state=origin_node, terminal_state=target_node)
    for _ in range(20):
        env.reset()
        neighbors = list(env.current_graph[env.current_state].keys())
        print(f"Neighbors of {env.current_state}: {neighbors}")
        action = random.choice(neighbors)
        state = env.start_state
        done = False
        while not done:
            neighbors = list(env.current_graph[state].keys())
            print(f"Neighbors of {state}: {neighbors}")
            # Selección de una acción aleatoria
            action = random.choice(list(env.current_graph[state].keys()))
            state, reward, done, info = env.take_action(state, action)
            print(f"State: {state}, Reward: {reward}, Done: {done}")
            print(info)
