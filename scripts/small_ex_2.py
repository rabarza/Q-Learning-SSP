import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))  # noqa: E402
from RLib.action_selection.action_selector import (
    EpsilonGreedyActionSelector,
    DynamicEpsilonGreedyActionSelector,
    UCB1ActionSelector,
    Exp3ActionSelector,
)
from RLib.utils.tables import dict_states_actions_zeros
from RLib.utils.files import download_graph, save_model_results
from RLib.utils.serializers import serialize_table
from RLib.utils.dijkstra import (
    get_optimal_policy,
    get_shortest_path_from_policy,
    get_q_table_for_policy,
    get_q_table_for_path,
    
)
from RLib.agents.ssp import QAgentSSP
from RLib.environments.ssp import SSPEnv
import numpy as np
import networkx as nx
import osmnx as ox
import winsound  # Para hacer sonar un beep al finalizar el entrenamiento
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")


print("Descargando datos y creando el grafo...")
location_point = (33.299896, -111.831638)
G = ox.graph_from_point(
    location_point, network_type="drive", dist=300, simplify=False
)  # 120
G = ox.utils_graph.get_largest_component(G, strongly=True)

# Seleccionar el nodo de origen y el nodo de destino
orig_node = list(G.nodes())[6]
dest_node = list(G.nodes())[65]
route = ox.shortest_path(G, orig_node, dest_node, weight="length")
# ox.plot_graph_route(G, route, node_size=0, edge_linewidth=0.5, figsize=(5,5))

if not nx.is_strongly_connected(G):
    G = ox.utils_graph.get_largest_component(G, strongly=True)

print(f"Nodo más cercano al punto de origen: {orig_node}")
print(f"Nodo más cercano al punto de destino: {dest_node}")
assert G.has_node(orig_node), "El nodo de origen no está en el grafo."
assert G.has_node(dest_node), "El nodo de destino no está en el grafo."

# ==================================================================================================
# Nodo de inicio y nodo de destino
inicio = orig_node
target = dest_node

# Calcular el camino más corto desde el nodo de inicio al nodo de destino
distancias, padres, shortest_path = dijkstra_shortest_path(
    G, inicio, target, logNormalExpectation=True, get_shortest_path=True
)
# Obtener la política óptima a partir del camino más corto
policy = get_path_as_stateactions_dict(shortest_path)
# Obtener la tabla Q óptima a partir de la política óptima
q_star = get_qtable_for_semipolicy(G, policy, target)

environment = SSPEnv(grafo=G, start_state=orig_node, terminal_state=dest_node)
NUM_EPISODES = 1000


methods = ["e-greedy", "UCB1", "exp3"]
expressions = ["\log t", "\sqrt{t}"]

for method in methods:
    print(f"Comenzando entrenamiento con método {method}")
    start_time = time.time()

    distribution = "lognormal"
    model = QAgentSSP(environment, strategy=method, epsilon=0.1, alpha=0.9, gamma=1)
    model.train(
        NUM_EPISODES,
        verbose=False,
        distribution=distribution,
        policy=policy,
        q_star=q_star,
    )

    model.iteration_time = time.time() - start_time
    save_model_results(model)

    ox.plot_graph_route(
        G,
        model.best_path(orig_node),
        node_size=0,
        edge_linewidth=0.5,
        figsize=(5, 5),
        save=True,
        filepath=f"results/imgs/{distribution}/{method}_shortest_path.png",
    )
