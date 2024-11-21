import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))  # noqa: E402
import networkx as nx
from RLib.cost_distributions import expected_time
from RLib.environments.ssp import get_edge_length, get_edge_speed
from RLib.utils.tables import dict_states_actions_zeros
from stqdm import stqdm
from tqdm import tqdm
from typing import List, Dict, Tuple, Any

# ======================= Dijkstra =======================


def dijkstra_shortest_path(graph, source, target, avg_speed=25, distribution="lognormal"):
    """
    Calcula el camino más corto desde el nodo de origen al nodo de destino usando el algoritmo de Dijkstra. Minimizando el tiempo de viaje de acuerdo a la velocidad promedio de los vehículos en la ciudad del grafo. Si se usa la esperanza de la distribución log-normal del peso escogido ('weight') como peso de los arcos, se minimiza el tiempo de viaje esperado.

    Parameters
    ----------
    graph: networkx.classes.multidigraph.MultiDiGraph
        grafo de la ciudad de Santiago
    source: int
        nodo de origen
    target: int
        nodo de destino
    avg_speed: int
        velocidad promedio de los vehículos en la ciudad del grafo

    Returns
    -------
    shortest_distances: dict
        diccionario con las distancias más cortas desde el nodo de origen. Tiene la forma {nodo: distancia, ..., nodo: distancia}
    parents: dict
        diccionario con los padres de cada nodo en el camino más corto. Tiene la forma {nodo: padre, ..., nodo: padre}
    shortest_path: list
        lista con el camino más corto desde el nodo de origen al nodo de destino. Tiene la forma [nodo, ..., nodo]
    """
    # Crear un diccionario para almacenar las distancias más cortas desde el nodo de origen
    shortest_distances = {node: float("inf") for node in graph.nodes()}
    shortest_distances[source] = 0  # La distancia al nodo de origen es 0
    # Crear un diccionario para almacenar los padres de cada nodo en el camino más corto
    parents = {node: None for node in graph.nodes()}
    # Lista de nodos no visitados
    unvisited_nodes = list(graph.nodes())
    while unvisited_nodes:
        # Seleccionar el nodo con la distancia más corta (nodo provisional)
        current_node = min(
            unvisited_nodes, key=lambda node: shortest_distances[node])
        # Eliminar el nodo actual de la lista de no visitados
        unvisited_nodes.remove(current_node)
        # Si alcanzamos el nodo de destino, podemos terminar el algoritmo
        if current_node == target:
            break
        # Calcular las distancias más cortas para los vecinos del nodo actual
        for neighbor in graph.neighbors(current_node):
            # Calcular la distancia desde el nodo actual al vecino
            arc_length = get_edge_length(graph, current_node, neighbor)
            arc_speed = get_edge_speed(
                graph, current_node, neighbor, avg_speed)

            time = expected_time(arc_length, arc_speed, distribution)

            distance = shortest_distances[current_node] + time

            if distance < shortest_distances[neighbor]:
                shortest_distances[neighbor] = distance
                parents[neighbor] = current_node

    shortest_path = get_shortest_path_from_parents(
        parents, source, target) if target else None
    return shortest_distances, parents, shortest_path


def get_shortest_path_from_parents(parents, source, target):
    """Retorna el camino más corto desde el nodo de origen al nodo de destino.
    Parameters
    ----------
    parents: dict
        diccionario con los padres de cada nodo en el camino más corto. Tiene la forma {nodo: padre, ..., nodo: padre}
    source: int
        nodo de origen
    target: int
        nodo de destino

    Returns
    -------
    shortest_path: list
        lista con el camino más corto desde el nodo de origen al nodo de destino. Tiene la forma [nodo, ..., nodo]
    """
    nodo_actual = target
    shortest_path = [nodo_actual]
    while nodo_actual != source:
        nodo_actual = parents[nodo_actual]
        shortest_path.append(nodo_actual)
    shortest_path.reverse()
    return shortest_path


def get_optimal_policy_and_q_star(graph: nx.MultiDiGraph, dest_node: Any, distribution: str = "lognormal", st: bool = False) -> Tuple[Dict[Any, Any], Dict[Any, Dict[Any, float]]]:
    """
    Calcula la política óptima y la tabla Q* realizando una búsqueda de Dijkstra desde el nodo de destino.

    Parameters
    ----------
    graph: networkx.classes.multidigraph.MultiDiGraph
        grafo de una ciudad
    dest_node: int
        nodo de destino

    Returns
    -------
    policy: dict
        Diccionario con la política óptima para cada nodo del grafo. Tiene la forma {nodo: acción, ..., nodo: acción}.
    Q_star: dict
        Diccionario con la tabla Q*. Tiene la forma {nodo: {acción: costo, ..., acción: costo}, ..., nodo: {...}}.
    """
    # Invertir el grafo
    graph_reversed = graph.reverse(copy=True)

    # Realizar Dijkstra desde el nodo de destino en el grafo invertido
    distancias, padres, _ = dijkstra_shortest_path(
        graph_reversed, dest_node, None, distribution=distribution
    )

    # Inicializar la tabla Q* y la política óptima
    Q_star = {}
    policy = {}

    # Barra de progreso (loading bar)
    progress_bar = stqdm(graph.nodes(), desc="Calculando tabla Q y política óptima") if st else tqdm(
        graph.nodes(), desc="Calculando tabla Q y política óptima")

    # Iterar sobre todos los nodos
    for s in progress_bar:
        Q_star[s] = {}
        best_action = None
        best_cost = float('inf')

        for a in graph.neighbors(s):
            # El costo de (s, a) es el costo del arco más el costo óptimo desde 'a' hasta el destino
            arc_length = get_edge_length(graph, s, a)
            arc_speed = get_edge_speed(graph, s, a)
            time = expected_time(arc_length, arc_speed, distribution)
            total_cost = time + distancias.get(a, float('inf'))

            # Guardar el costo en la tabla Q*
            Q_star[s][a] = -total_cost

            # Determinar la mejor acción para la política óptima
            if total_cost < best_cost:
                best_cost = total_cost
                best_action = a

        # La política óptima en el estado s es la acción con el menor costo en Q*
        policy[s] = best_action

    # En el nodo de destino, la acción óptima es quedarse en el destino
    policy[dest_node] = dest_node
    Q_star[dest_node] = {dest_node: 0}

    return policy, Q_star


def get_shortest_path_from_policy(policy, source, target):
    """Retorna el camino más corto desde el nodo de origen al nodo de destino a partir de la política óptima.

    Parameters
    ----------
    policy: dict
        Diccionario que contiene la política óptima para el camino más corto entre el nodo de inicio y el nodo de destino.
    source: int
        nodo de origen
    target: int
        nodo de destino

    Returns
    -------
    path: list
        lista con el camino más corto desde el nodo de origen al nodo de destino. Tiene la forma [nodo, ..., nodo]
    """

    path = [source]
    node = source
    while node != target:
        # Siguiente nodo en el camino más corto
        node = policy[node]
        # Agregar el nodo al camino
        path.append(node)
    return path


def get_path_as_stateactions_dict(path):
    """
    Retorna el camino `path` como diccionario de estados acciones. Se define como la política a seguir para los nodos del camino más corto (no para todos los nodos del grafo).

    Parameters
    ----------
    path : list
        Lista de nodos que conforman el camino. más corto entre el nodo de inicio y el nodo de destino.
    Returns
    -------
    policy : dict
        Diccionario que contiene la política óptima para el camino más corto entre el nodo de inicio y el nodo de destino. Tiene la forma {nodo: acción, ..., nodo: acción} donde la acción es el siguiente nodo en el camino más corto.

    """
    states = path[:-1]
    actions = path[1:]
    path_dict = {state: action for state, action in zip(states, actions)}
    path_dict[path[-1]] = path[-1]
    return path_dict


def get_qtable_for_semipolicy(graph, policy, dest):
    """
    Retorna la tabla Q óptima para la política óptima.

    Parameters
    ----------
    policy : dict
        Diccionario que contiene la política óptima para el camino más corto entre el nodo de inicio y el nodo de destino.
    distances : dict
        Diccionario que contiene las distancias de cada nodo al nodo de destino.
    Returns
    -------
    q_table : dict
        Diccionario que contiene la tabla Q óptima para la política óptima.
    """
    from RLib.environments.ssp import get_cumulative_edges_cost

    q_table = dict_states_actions_zeros(graph)
    for key, value in policy.items():
        # costo acumulado de ir desde el nodo 'key' al nodo de destino 'dest' siguiendo la política 'policy'
        q_table[key][value] = - \
            get_cumulative_edges_cost(graph, policy, key, dest)
    return q_table


def get_q_table_for_path(q_table: Dict[Any, Dict[Any, float]], path: List[Any]) -> Dict[Any, Dict[Any, float]]:
    """Dado un camino y una tabla Q,  retorna la tabla Q con los pares estado-acción restringidos a sólo los nodos del camino, es decir, sólo los nodos que conforman el camino más corto entre el nodo de inicio y el nodo de destino. 

    Parameters
    ----------
        q_table (Dict): tabla de valores Q
        path (List): camino sobre el cual se restringirá la tabla Q

    Returns:
    -------
        Dict[Any, Dict[Any, float]]: Tabla Q restringida a los nodos del camino
    """
    q_table_for_path = {}
    for i in range(len(path) - 1):
        node = path[i]
        next_node = path[i + 1]
        q_table_for_path[node] = {next_node: q_table[node][next_node]}
    q_table_for_path[path[-1]] = {path[-1]: 0}
    return q_table_for_path


if __name__ == '__main__':

    # Ejemplo de calculo de política óptima y tabla Q*
    grafo = nx.MultiDiGraph()

    # Añadir nodos y aristas con pesos al grafo
    edges = [
        (0, 1, {'length': 10, 'speed_kph': 25}),
        (0, 2, {'length': 15, 'speed_kph': 30}),
        (1, 3, {'length': 12, 'speed_kph': 25}),
        (2, 3, {'length': 10, 'speed_kph': 30}),
        (1, 2, {'length': 5, 'speed_kph': 35}),
        (3, 4, {'length': 7, 'speed_kph': 20}),
    ]

    grafo.add_edges_from(edges)

    target_node = 4
    costs_distribution = "uniform"
    policy, optimal_q_table = get_optimal_policy_and_q_star(
        grafo, target_node, costs_distribution)
    print("Política óptima:")
    print(policy)
    print("Tabla Q*:")
    print(optimal_q_table)
