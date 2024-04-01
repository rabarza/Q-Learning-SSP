import osmnx as ox
import networkx as nx
import numpy as np
from RLib.environments.ssp import SSPEnv
from RLib.agents.ssp import QAgentSSP
from RLib.utils.dijkstra_utils import (
    dijkstra_shortest_path,
    get_path_as_stateactions_dict,
    get_qtable_for_semipolicy,
    get_q_table_for_policy,
    get_optimal_policy,
)
from RLib.utils.file_utils import download_graph, save_model_results
from RLib.utils.table_utils import dict_states_actions_zeros
import winsound  # Para hacer sonar un beep al finalizar el entrenamiento
from RLib.action_selection.action_selector import (
    EpsilonGreedyActionSelector,
    DynamicEpsilonGreedyActionSelector,
    UCB1ActionSelector,
    Exp3ActionSelector,
)

# Descargar el grafo
G = ox.graph_from_place("Piedmont, California", network_type="drive")
G = ox.utils_graph.get_largest_component(
    G, strongly=True
)  # Obtener la componente fuertemente conexa más grande
# fig, ax = ox.plot_graph(G, node_size=0, edge_linewidth=0.5, figsize=(10,10))

# # Nodos de origen y destino
orig_node = 53017091
dest_node = 53035699

# Calcular el camino más corto desde el nodo de inicio al nodo de destino
distancias, padres, shortest_path = dijkstra_shortest_path(G, orig_node, dest_node)

# Obtener la política óptima a partir del camino más corto dado el shortest_path
policy = get_path_as_stateactions_dict(shortest_path)

# Obtener la tabla Q óptima a partir de la política óptima
q_star = get_qtable_for_semipolicy(G, policy, dest_node)

# Obtener la política óptima para cada nodo en el grafo hasta el destino
policies = get_optimal_policy(G, dest_node)

# Obtener la tabla Q* a partir de las políticas óptimas
q_star = get_q_table_for_policy(G, policies, dest_node)

alpha_values = np.linspace(0.01, 0.1, 10)
NUM_EPISODES = 40000
# Crear un entorno
env = SSPEnv(grafo=G, start_state=orig_node, terminal_state=dest_node)

greedy_agents = []

selectors = {
    "e-greedy": EpsilonGreedyActionSelector,
    "UCB1": UCB1ActionSelector,
    "exp3": Exp3ActionSelector,
}

strategies = ["e-greedy", "UCB1", "exp3"]

dynamic_alpha = False
for strategy in strategies:
    if not dynamic_alpha:
        for alpha in alpha_values:
            piedmont_eps = QAgentSSP(env, alpha=alpha, gamma=1)
            piedmont_eps.train(
                NUM_EPISODES, q_star=q_star, policy=policy, distribution="lognormal"
            )
            greedy_agents.append(piedmont_eps)
            name = f"piedmont_eps_{alpha:.2f}"
            save_model_results(
                piedmont_eps,
                nombre=name,
                path=f"results/piedmont/{strategy}/constant_alpha/",
            )
    else:
        piedmont_eps = QAgentSSP(env, alpha=alpha, gamma=1, dynamic_alpha=True)
        piedmont_eps.train(
            NUM_EPISODES, q_star=q_star, policy=policy, distribution="lognormal"
        )
        greedy_agents.append(piedmont_eps)
        name = f"piedmont_eps_dynamic_alpha"
        save_model_results(
            piedmont_eps,
            nombre=name,
            path=f"results/piedmont/dynamic_alpha/{strategy}/",
        )

winsound.Beep(2000, 4000)  # Beep con frecuencia de 1000 Hz durante 2 segundos
