from typing import Tuple, Dict, Any
import random
import numpy as np
import networkx as nx
import osmnx as ox
import copy
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from RLib.distributions.distributions import (
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
        self.__start_state = start_state
        self.__current_state = self.__start_state
        assert terminal_state is not None, "El estado terminal no puede ser None"
        self.__terminal_state = terminal_state
        self.__num_states = graph.number_of_nodes()
        self.__num_actions = {
            k: v
            for k, v in map(
                lambda item: (item[0], len(item[1])
                              ), nx.to_dict_of_lists(graph).items()
            )
        }  # Diccionario con el número de acciones por estado
        self.__adjacency_dict_of_lists = nx.to_dict_of_lists(self.graph)
        self.__q_table = self.dict_states_actions_zeros()
        self.__costs_distribution = costs_distribution
        self.__shortest_path = shortest_path

    def __str__(self):
        return f"EnvShortestPath {self.graph}"

    @property
    def num_states(self) -> int:
        return self.__num_states

    @property
    def num_actions(self) -> Dict[Any, int]:
        return self.__num_actions

    @property
    def start_state(self) -> Any:
        return self.__start_state

    @property
    def terminal_state(self) -> Any:
        return self.__terminal_state

    @property
    def current_state(self) -> Any:
        return self.__current_state

    @property
    def costs_distribution(self) -> str:
        return self.__costs_distribution

    @property
    def shortest_path(self) -> list:
        return self.__shortest_path

    def reset(self):
        """Reiniciar el entorno de aprendizaje"""
        self.__current_state = self.__start_state
        print(
            f"Entorno reiniciado. Grafo con {self.num_nodos} nodos. Estado inicial: {self.__start_state}. Estado final: {self.__terminal_state}"
        )

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
            state in self.__adjacency_dict_of_lists.keys()
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
                             self.__costs_distribution)
        terminated = self.check_state(next_state)
        # La recompensa es negativa porque se busca maximizar la recompensa que representa minimizar el costo
        reward = - cost
        self.__current_state = next_state
        info = str({"estado": state, "recompensa": reward, "terminado": terminated})
        return next_state, reward, terminated, info

    def dict_states_actions_zeros(self) -> Dict[str, Dict[str, float]]:
        """Crear un diccionario con estados y acciones con valores 0"""
        return dict_states_actions_zeros(self.graph)

    def dict_states_actions_constant(self, constant) -> Dict[str, Dict[str, float]]:
        table = dict_states_actions_constant(self.graph, constant)
        table[self.terminal_state] = {self.terminal_state: 0}
        return table
    
    def dict_states_zeros(self) -> Dict[str, float]:
        """Crear un diccionario con estados con valores 0"""
        return dict_states_zeros(self.graph)

    def calculate_optimal_qtable(self):
        """### Calcular la tabla Q óptima"""
        pass


if __name__ == "__main__":
    pass
