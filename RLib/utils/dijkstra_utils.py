import networkx as nx
from RLib.distributions.distributions import expected_time, expected_speed
from RLib.environments.ssp import get_edge_length, get_edge_speed
from RLib.utils.table_utils import dict_states_actions_zeros
from stqdm import stqdm


# ======================= Dijkstra =======================
def dijkstra_shortest_path(graph, source, target, avg_speed=25):
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
        current_node = min(unvisited_nodes, key=lambda node: shortest_distances[node])
        # Eliminar el nodo actual de la lista de no visitados
        unvisited_nodes.remove(current_node)
        # Si alcanzamos el nodo de destino, podemos terminar el algoritmo
        if current_node == target:
            break
        # Calcular las distancias más cortas para los vecinos del nodo actual
        for neighbor in graph.neighbors(current_node):
            # Calcular la distancia desde el nodo actual al vecino
            arc_length = get_edge_length(graph, current_node, neighbor)
            arc_speed = get_edge_speed(graph, current_node, neighbor, avg_speed)

            time = expected_time(arc_length, arc_speed)

            distance = shortest_distances[current_node] + time

            if distance < shortest_distances[neighbor]:
                shortest_distances[neighbor] = distance
                parents[neighbor] = current_node

    shortest_path = get_shortest_path(parents, source, target)
    return shortest_distances, parents, shortest_path


def get_shortest_path(parents, source, target):
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
    dest_node = path[-1]
    # Inicializar la política excluyendo el nodo de destino
    path_dict = {node: 0 for node in path if node != dest_node}
    for index in range(len(path) - 1):
        node = path[index]
        next_node = path[index + 1]
        path_dict[node] = next_node
    return path_dict


def get_optimal_policy(grafo, dest_node):
    """
    Calcula la política óptima completa para cada nodo del grafo realizando una búsqueda de Dijkstra desde cada nodo.

    Parameters
    ----------
    grafo: networkx.classes.multidigraph.MultiDiGraph
        grafo de una ciudad  
    dest_node: int
        nodo de destino

    Returns
    -------
    policies: dict
        diccionario con las políticas óptimas para cada nodo del grafo. Tiene la forma {nodo: política, ..., nodo: política}
    """
    # Iniciliazar el conjunto de nodos visitados
    visited_nodes = set()
    # Inicializar el conjunto de nodos restantes
    remaining_nodes = set(grafo.nodes())
    # Inicializar el diccionario de política
    policies = {}
    
    while remaining_nodes:
        # Seleccionar un nodo no visitado
        source_node = next(node for node in remaining_nodes)
        # Realizar una búsqueda de Dijkstra desde el nodo seleccionado como source_node
        distancias, padres, shortest_path = dijkstra_shortest_path(grafo, source_node, dest_node)
        shortest_path_as_dict = get_path_as_stateactions_dict(shortest_path)
        # Agrega todas las llaves y valores del diccionario shortest_path_as_dict al diccionario policies
        policies.update(shortest_path_as_dict)
        # Agraga todos los nodos visitados al conjunto de nodos visitados
        visited_nodes.update(shortest_path)
        # Elimina todos los nodos visitados del conjunto de nodos restantes
        remaining_nodes.difference_update(visited_nodes)
    return policies


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
        q_table[key][value] = -get_cumulative_edges_cost(graph, policy, key, dest)
    return q_table


def get_q_table_for_policy(graph, policy, dest_node):
    """Retorna la tabla Q óptima a partir de las políticas óptimas de cada nodo.
    Tiene la forma {estado: {accion: valor, ..., accion: valor}, ..., estado: {accion: valor, ..., accion: valor}}
    """
    from RLib.environments.ssp import get_edge_cost, get_cumulative_edges_cost

    q_star = dict_states_actions_zeros(graph)
    for state in stqdm(q_star.keys(), desc="Calculando tabla Q"):
        if state == dest_node:
            continue
        for action in q_star[state].keys():
            tij = get_edge_cost(
                graph, state, action
            )  # costo de ir desde el estado 'state' al estado 'action'

            # Obtener la política de todos los nodos del camino más corto desde el nodo 'state' al nodo de destino 'dest_node'
            q_star[state][action] = -(
                tij + get_cumulative_edges_cost(graph, policy, action, dest_node)
            )
    return q_star
