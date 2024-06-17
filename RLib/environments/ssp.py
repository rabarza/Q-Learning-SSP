import random
import numpy as np
import networkx as nx
import osmnx as ox
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from RLib.distributions.distributions import (
    LogNormalDistribution,
    expected_time,
    random_time,
)

##############################################################################################################
# Stochastic Shortest Path
##############################################################################################################
def get_edge_attribute(G, orig, dest, weight="length"):
    edge_data = G.get_edge_data(orig, dest)
    if weight in edge_data:
        edge_length = edge_data[weight]
    else:
        edge_features = next(iter(edge_data.values()))
        assert (
            weight in edge_features
        ), f"No hay atributo {weight} en los arcos entre los nodos {orig} y {dest}."
        edge_length = edge_features[weight]
    return edge_length


def get_edge_length(G, orig, dest):
    assert G.has_edge(orig, dest), f"No hay arco entre los nodos {orig} y {dest}."
    return get_edge_attribute(G, orig, dest, "length")


def get_edge_speed(G, orig, dest, avg_speed=25):
    try:
        # edge_speed = get_edge_attribute(G, orig, dest, "speed_kph")
        edge_speed = avg_speed
    except Exception:
        edge_speed = avg_speed
    if edge_speed - avg_speed > 10:
        edge_speed = avg_speed
    return edge_speed


def get_edge_cost(
    G,
    origen: int,
    destino: int,
    distribution_name: str = "expectation-lognormal",
    avg_speed=25,
):
    """### Obtener el costo de un arco entre dos nodos
    Parámetros:
        G (Grafo): Grafo con el que se va a trabajar
        origen (int): Nodo origen
        destino (int): Nodo destino
        distribution_name (str): Distribución de probabilidad que se va a utilizar para obtener el costo estocástico. Si no se especifica, el costo es determinístico y se utiliza el valor de la esperanza de la distribución lognormal.
    Retorno:
        stochastic_cost (float): Costo del arco entre los nodos origen y destino
    """
    edge_length = get_edge_length(G, origen, destino)
    edge_speed = get_edge_speed(G, origen, destino, avg_speed)

    # stochastic cost sample from lognormal distribution_name
    if distribution_name == "expectation-lognormal":
        # get the expected speed and calculate the expected time given the edge length
        time = expected_time(edge_length, edge_speed)

    elif distribution_name == "lognormal":
        # generate a sample of the speed and calculate the time given the edge length
        time = random_time(edge_length, edge_speed)
    else:
        raise ValueError(f"La distribución {distribution_name} no está implementada.")
    stochastic_cost = time
    return stochastic_cost


def get_cumulative_edges_cost(grafo, policy, node, dest_node):
    """
    Retorna el costo acumulado de comenzar en el nodo 'node' y llegar al nodo 'dest_node' siguiendo la política 'policy'
    """
    # Obtener el camino más corto desde el nodo de inicio al nodo de destino dada la política
    path = [node]
    while node != dest_node:
        node = policy[node]
        path.append(node)
    # Calcular el costo acumulado del camino más corto
    cost = 0
    for index in range(len(path) - 1):
        cost += get_edge_cost(grafo, path[index], path[index + 1])
    return cost


class SSPEnv:
    """
    Entorno de aprendizaje para el problema de encontrar el camino más corto en un grafo
    """

    def __init__(self, grafo, start_state, terminal_state):
        """Constructor de la clase SSPEnv
        
        Parámetros:
        ----------
        grafo (nx.Graph): Grafo con el que se va a trabajar
        start_state (int): Estado inicial
        terminal_state (int): Estado terminal
        """

        assert grafo is not None, "El grafo no puede ser None"
        self.grafo = grafo
        self.num_nodos = grafo.number_of_nodes()
        assert start_state is not None, "El estado inicial no puede ser None"
        self.__start_state = start_state
        self.__current_state = self.__start_state
        assert terminal_state is not None, "El estado terminal no puede ser None"
        self.__terminal_state = terminal_state
        self.__num_states = grafo.number_of_nodes()
        self.__num_actions = {
            k: v
            for k, v in map(
                lambda item: (item[0], len(item[1])), nx.to_dict_of_lists(grafo).items()
            )
        }  # Diccionario con el número de acciones por estado
        self.__adjacency_dict_of_lists = nx.to_dict_of_lists(self.grafo)
        self.__q_table = self.dict_states_actions_zeros()

    def __str__(self):
        return f"EnvShortestPath {self.grafo}"

    @property
    def num_states(self):
        return self.__num_states

    @property
    def num_actions(self):
        return self.__num_actions

    @property
    def start_state(self):
        return self.__start_state

    @property
    def terminal_state(self):
        return self.__terminal_state

    @property
    def current_state(self):
        return self.__current_state

    def dict_states_zeros(self):
        """Retorna un diccionario con los estados con valor 0. Es útil para inicializar la tabla del número de visitas a cada estado, por ejemplo. Tiene la forma {estado: 0, ..., estado: 0}"""
        return {state: 0 for state, actions in nx.to_dict_of_lists(self.grafo).items()}

    def dict_states_actions_zeros(self):
        """Retorna un diccionario con los estados y acciones con valor 0. Es útil para inicializar la tabla Q, o la tabla de la cantidad de veces que se ha visitado cada par estado-acción. Tiene la forma {estado: {accion: 0, ..., accion: 0}, ..., estado: {accion: 0, ..., accion: 0}}"""
        G = self.grafo
        return {
            state: {action: 0 for action in actions}
            for state, actions in nx.to_dict_of_lists(G).items()
        }

    def dict_states_actions_random(self):
        """Retorna un diccionario con los estados y acciones con valor aleatorio. Es útil para inicializar la tabla Q, o la tabla de la cantidad de veces que se ha visitado cada par estado-acción. Tiene la forma {estado: {accion: valor_aleatorio, ..., accion: valor_aleatorio}, ..., estado: {accion: valor_aleatorio, ..., accion: valor_aleatorio}}"""
        G = self.grafo
        return {
            state: {action: np.random.random() for action in actions}
            for state, actions in nx.to_dict_of_lists(G).items()
        }

    def reset(self):
        """Reiniciar el entorno de aprendizaje"""
        self.__current_state = self.__start_state
        print(
            f"Entorno reiniciado. Grafo con {self.num_nodos} nodos. Estado inicial: {self.__start_state}. Estado final: {self.__terminal_state}"
        )

    def check_state(self, state):
        """### Verificar si un estado es válido y/o terminal
        Parámetros:
            state (int): Estado a verificar
        Retorno:
            valid (bool): Indica si el estado es terminal y arroja un error si no es válido
        """
        assert (
            state in self.__adjacency_dict_of_lists.keys()
        ), f"El estado {state} no está en el grafo"
        return state == self.terminal_state

    def take_action(self, state, action, distribution="expectation-lognormal"):
        """### Tomar una acción en el entorno de aprendizaje
        Parámetros:
            state (int): Estado actual
            action (int): Acción a tomar
        Retorno:
            next_state (int): Siguiente estado
            reward (float): Recompensa (o costo) por tomar la acción
            terminated (bool): Indica si el episodio terminó
        """
        # Obtener los arcos del nodo actual (estado actual)
        assert self.grafo.has_edge(
            state, action
        ), f"La acción {action} no está en los arcos del nodo {state}"
        next_state = action
        cost = get_edge_cost(self.grafo, state, next_state, distribution)
        terminated = self.check_state(next_state)
        reward = -cost
        self.__current_state = next_state
        info = {"estado": state, "recompensa": reward, "terminado": terminated}
        return next_state, reward, terminated, info

    def calculate_optimal_qtable(self):
        """### Calcular la tabla Q óptima"""
        pass


if __name__ == "__main__":
    pass
