import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import copy
import osmnx as ox
import networkx as nx
import numpy as np
from RLib.environments.ssp import SSPEnv
from RLib.agents.ssp import QAgentSSP
from RLib.utils.dijkstra_utils import (
    get_optimal_policy,
    get_shortest_path_from_policy,
    get_q_table_for_policy,
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
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")


# Descargar el grafo y obtener el componente conexo más grande
location_name = "Piedmont, California"
G = ox.graph_from_place(location_name, network_type="drive")
# Añadir el atributo de velocidad a las aristas (speed_kph)
G = ox.add_edge_speeds(G)
G.to_directed()
G = ox.utils_graph.get_largest_component(
    G, strongly=True
)
# fig, ax = ox.plot_graph(G, node_size=0, edge_linewidth=0.5, figsize=(10,10))

# # Nodos de origen y destino
orig_node = 53017091
dest_node = 53035699
distribution = "normal"

# Añadir un arco recurrente de largo 0 en el nodo terminal
# Verificar si el borde entre dest_node y dest_node existe
if G.has_edge(dest_node, dest_node):
    # Actualizar los atributos del borde existente
    G[dest_node][dest_node][0]['length'] = 0  # Actualiza el atributo 'length'
else:
    # Agregar un nuevo borde con los atributos deseados
    G.add_edge(dest_node, dest_node, length=0)

# Calcular el camino más corto desde el nodo de inicio al nodo de destino
optimal_policy = get_optimal_policy(G, dest_node, distribution)
shortest_path = get_shortest_path_from_policy(
    optimal_policy, orig_node, dest_node
)

# Obtener la tabla Q* a partir de las políticas óptimas
optimal_q_table = get_q_table_for_policy(G, optimal_policy, dest_node, distribution, False)
# serialized_q_table = serialize_table(optimal_q_table)

NUM_EPISODES = 40000
# Crear un entorno
env = SSPEnv(G, orig_node, dest_node, distribution)


selectors = {
    "e-greedy": EpsilonGreedyActionSelector(epsilon=0.1),
    "UCB1": UCB1ActionSelector(c=2),
    "exp3": Exp3ActionSelector(eta="log(t)"),
}


alpha_type_dict = {"constante": "constant_alpha",
                   "dinámico": "dynamic_alpha"}

agents = []
dynamic_alpha = False
strategies = list(selectors.keys())
for strategy in strategies:
    # Crear el agente
    agent = QAgentSSP(env,
                      dynamic_alpha=dynamic_alpha,
                      alpha_formula="1 / N(s,a)",
                      action_selector=selectors[strategy]
                      )
    # Entrenar el agente
    agent.train(
        NUM_EPISODES,
        shortest_path=shortest_path,
        q_star=optimal_q_table,
    )
    agents.append(agent)

    for element in strategies:
        for alpha_type in list(alpha_type_dict.values()):
            temp_path = f"results/{location_name}/{orig_node}-{dest_node}/{alpha_type}/{element}/"
            results_dir = os.path.join(BASE_DIR, temp_path)
            # Si no existe la carpeta, crearla
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)

    alpha_type_dir = alpha_type_dict[alpha_type]
    # Ruta para guardar resultados
    agent_storage_path = os.path.join(
        BASE_DIR,
        "results/",
        f"{location_name}/{orig_node}-{dest_node}/{alpha_type_dir}/{strategy}/",
    )

    # Si no existe la carpeta, crearla
    if not os.path.exists(agent_storage_path):
        os.makedirs(agent_storage_path)

    # Guardar resultados
    save_model_results(
        agent, nombre=f"QAgentSSP_", path=agent_storage_path
    )

winsound.Beep(2000, 4000)  # Beep con frecuencia de 1000 Hz durante 2 segundos
