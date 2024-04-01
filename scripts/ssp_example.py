import osmnx as ox
import networkx as nx
import numpy as np
import time
from RLib.environments.ssp import SSPEnv
from RLib.agents.ssp import QAgentSSP
from RLib.utils.dijkstra_utils import (
    dijkstra_shortest_path,
    get_path_as_stateactions_dict,
    get_qtable_for_semipolicy,
)
from RLib.utils.table_utils import max_norm
from RLib.utils.file_utils import save_model_results, download_graph
from RLib.utils.table_utils import resta_diccionarios


# # Aplicación del algoritmo Q-Learning para encontrar el camino más corto entre dos puntos de la ciudad de Santiago
print("Descargando datos y creando el grafo...")
G = download_graph(filepath="data/santiago_connected")
if not nx.is_strongly_connected(G):
    G = ox.utils_graph.get_largest_component(G, strongly=True)

assert nx.is_strongly_connected(G), "El grafo no es fuertemente conexo."

# Añadir atributo de velocidad
# G = ox.add_edge_speeds(G)
# fig, ax = ox.plot_graph(G, node_color='r', node_size=1, save=True, filepath='graph.png')

# # Nodos de origen y destino Comuna de Santiago, Chile (longitud, latitud)
orig_node = ox.distance.nearest_nodes(G, X=-70.67278, Y=-33.47492)
dest_node = ox.distance.nearest_nodes(G, X=-70.6451, Y=-33.4356)

print(f"Nodo más cercano al punto de origen: {orig_node}")
print(f"Nodo más cercano al punto de destino: {dest_node}")
assert G.has_node(orig_node), "El nodo de origen no está en el grafo."
assert G.has_node(dest_node), "El nodo de destino no está en el grafo."


# Calcular el camino más corto desde el nodo de inicio al nodo de destino
distancias, padres, shortest_path = dijkstra_shortest_path(
    G, orig_node, dest_node, logNormalExpectation=True, get_shortest_path=True
)
# Obtener la política óptima a partir del camino más corto
policy = get_path_as_stateactions_dict(shortest_path)
# Obtener la tabla Q óptima a partir de la política óptima
q_star = get_qtable_for_semipolicy(G, policy, dest_node)


environment = SSPEnv(grafo=G, start_state=orig_node, terminal_state=dest_node)
NUM_EPISODES = 15000

# epsilons = [0.1, 0.2, 0.3, 0.4, 0.5]
# cs = [1, 2, 3, 4, 5]
# methods = ['e-greedy', 'UCB1', 'exp3']
methods = ["e-greedy"]

# methods = ['e-greedy', 'exp3']
expressions = ["\log t", "\sqrt{t}"]

for method in methods:
    print(f"Comenzando entrenamiento con método {method}")

    # start_time = time.time()
    # distribution = "expectation-lognormal"
    # model = QAgentSSP(environment, strategy=method, epsilon=0.1, alpha=0.1, gamma=1)
    # model.train(NUM_EPISODES, verbose=False, distribution=distribution, policy=policy, q_star=q_star)
    # model.iteration_time = time.time() - start_time
    # save_model_results(model)

    start_time = time.time()
    distribution = "lognormal"
    model = QAgentSSP(environment, strategy=method, epsilon=0.1, alpha=0.1, gamma=1)
    model.train(
        NUM_EPISODES,
        verbose=False,
        distribution=distribution,
        policy=policy,
        q_star=q_star,
    )
    model.iteration_time = time.time() - start_time
    save_model_results(model)

    # ox.plot_graph_route(G, model.best_path(orig_node), node_size=0, edge_linewidth=0.5, figsize=(5,5), save=True, filepath=f"results/imgs/{distribution}/{method}_shortest_path.png")
resta = resta_diccionarios(model.q_table, q_star)
# print(resta)
for estado, accion in policy.items():
    print(
        f"{(estado, accion)} {(model.q_table[estado][accion] - q_star[estado][accion])} {model.q_table[estado][accion]} {q_star[estado][accion]} "
    )
