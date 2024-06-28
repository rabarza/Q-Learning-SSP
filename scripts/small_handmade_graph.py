import osmnx as ox
import networkx as nx
import time
from RLib.environments.ssp import SSPEnv
from RLib.agents.ssp import QAgentSSP
from RLib.utils.dijkstra import (
    dijkstra_shortest_path,
    get_path_as_stateactions_dict,
    get_qtable_for_semipolicy,
)
from RLib.utils.tables import (
    dict_states_actions_zeros,
    resta_diccionarios,
    max_norm,
    max_value_in_dict,
)
from RLib.utils.files import save_model_results, load_model_results


# Crear un grafo dirigido
grafo = nx.DiGraph()

# Agregar nodos al grafo
nodos = [1, 2, 3, 4, 5, 6]
grafo.add_nodes_from(nodos)

# aristas y sus atributo de longitud
aristas = [
    (1, 2, {"length": 30}),
    (2, 3, {"length": 22}),
    (3, 4, {"length": 12}),
    (4, 5, {"length": 11}),
    (5, 6, {"length": 14}),
    (6, 1, {"length": 7}),
    (2, 5, {"length": 9}),
    (3, 6, {"length": 2}),
]

# Agregar las aristas al grafo con el atributo del largo
grafo.add_edges_from(aristas)

# Posición de los nodos en el gráfico (ayuda a graficar)
pos = nx.spring_layout(grafo)

# Etiquetar los arcos con el largo
edge_labels = {(u, v): d["length"] for u, v, d in grafo.edges(data=True)}

# Graficar
nx.draw_networkx(
    grafo,
    pos,
    with_labels=True,
    node_size=700,
    node_color="skyblue",
    arrows=True,
    node_shape="o",
    font_size=12,
    font_weight="bold",
)
nx.draw_networkx_edge_labels(grafo, pos, edge_labels=edge_labels)

orig_node = 1
dest_node = 6

route = ox.shortest_path(grafo, orig_node, dest_node, weight="length")
# fig, ax = ox.plot_graph_route(grafo, route ,node_size=0, edge_linewidth=0.5, figsize=(5,5))

if not nx.is_strongly_connected(grafo):
    grafo = ox.utils_graph.get_largest_component(grafo, strongly=True)

assert nx.is_strongly_connected(
    grafo
), "El grafo no es fuertemente conexo. Asegúrate de que todos los nodos estén conectados."

assert grafo.has_node(orig_node), "El nodo de origen no está en el grafo."
assert grafo.has_node(dest_node), "El nodo de destino no está en el grafo."

# ==================================================================================================
# Nodo de inicio y nodo de destino
inicio = orig_node
target = dest_node

# Calcular el camino más corto desde el nodo de inicio al nodo de destino
distancias, padres, shortest_path = dijkstra_shortest_path(
    grafo, inicio, target, logNormalExpectation=True, get_shortest_path=True
)

# Obtener la política óptima a partir del camino más corto
policy = get_path_as_stateactions_dict(shortest_path)

# Obtener la tabla Q óptima a partir de la política óptima
q_star = get_qtable_for_semipolicy(grafo, policy, target)

environment = SSPEnv(grafo=grafo, start_state=orig_node, terminal_state=dest_node)
NUM_EPISODES = 500
method = "UCB1"

print(f"Comenzando entrenamiento con método {method}")
start_time = time.time()

model = QAgentSSP(environment, strategy=method, epsilon=0.1, alpha=0.1, gamma=1)
model.train(
    NUM_EPISODES, verbose=False, distribution="lognormal", policy=policy, q_star=q_star
)
iteration_time = time.time() - start_time
model.iteration_time = iteration_time
save_model_results(model)
total_time = time.time() - start_time
print(f"Tiempo de ejecución del entrenamiento: {iteration_time:.2f} segundos")
resta = resta_diccionarios(model.q_table, q_star)
print(resta)
