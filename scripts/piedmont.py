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


# Descargar el grafo y obtener el componente conexo más grande
location_name = "Piedmont, California"
G = ox.graph_from_place(location_name, network_type="drive")
# Añadir el atributo de velocidad a las aristas (speed_kph)
G = ox.add_edge_speeds(G)
G.to_directed()
# Define la longitud mínima (en metros, por ejemplo)
min_length = 3

# Filtrar los bordes que cumplen con el criterio de longitud mínima
edges_to_remove = [(u, v) for u, v, data in G.edges(data=True) if data.get('length', 0) < min_length]

# Utiliza la función remove_edges de osmnx para eliminar los bordes
G.remove_edges_from(edges_to_remove)

G = ox.truncate.largest_component(
    G, strongly=True
)  # Obtener el componente fuertemente conexo más grande
# Nodos de origen y destino
orig_node = 53017091
dest_node = 53035699
distribution = "normal"

# Añadir un arco recurrente de largo 0 en el nodo terminal
if G.has_edge(dest_node, dest_node):  # Si ya existe el arco
    G[dest_node][dest_node][0]['length'] = 0  # Actualiza el atributo 'length'
else:
    G.add_edge(dest_node, dest_node, length=0)

# Calcular el camino más corto desde el nodo de inicio al nodo de destino
optimal_policy = get_optimal_policy(G, dest_node, distribution)
shortest_path = get_shortest_path_from_policy(
    optimal_policy, orig_node, dest_node
)

# Obtener la tabla Q* a partir de las políticas óptimas
optimal_q_table = get_q_table_for_policy(
    G, optimal_policy, dest_node, distribution, False)
optimal_q_table_for_sp = get_q_table_for_path(optimal_q_table, shortest_path)


# Serializar q_star
serialized_opt_q_table = serialize_table(optimal_q_table)
serialized_opt_q_table_for_sp = serialize_table(optimal_q_table_for_sp)
# Convertir a formato JSON
json_q_star = json.dumps(serialized_opt_q_table, indent=4)
json_shortest_path = json.dumps(shortest_path, indent=4)
json_q_star_for_sp = json.dumps(serialized_opt_q_table_for_sp, indent=4)
# Guardar en un archivo
with open(os.path.join(RESULTS_DIR, f"q_star_Piedmont_distr_{distribution}.json"), "w") as f:
    f.write(json_q_star)
    f.write("\n")
    f.write(json_shortest_path)
    f.close()
with open(os.path.join(RESULTS_DIR, f"q_star_for_shortest_path_Piedmont_distr_{distribution}.json"), "w") as f:
    f.write(json_q_star_for_sp)
    f.close()

NUM_EPISODES = 40000
# Crear un entorno
env = SSPEnv(G, orig_node, dest_node, distribution)


selectors = {
    "e-greedy": EpsilonGreedyActionSelector(epsilon=0.1),
    "UCB1": UCB1ActionSelector(c=2),
    "exp3": Exp3ActionSelector(eta="log(t+1)"),
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
                      alpha_formula="0.1",
                      action_selector=selectors[strategy]
                      )
    # Entrenar el agente
    print(f"Training agent with strategy: {strategy}")
    agent.train(
        NUM_EPISODES,
        shortest_path=shortest_path,
        q_star=optimal_q_table,
    )
    agents.append(agent)

    for element in strategies:
        for alpha_type in list(alpha_type_dict.values()):
            temp_path = f"{location_name}/{orig_node}-{dest_node}/{alpha_type}/{element}/"
            results_dir = os.path.join(RESULTS_DIR, temp_path)
            # Si no existe la carpeta, crearla
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)

    # Ruta para guardar resultados
    agent_storage_path = os.path.join(
        BASE_DIR,
        "results/",
        f"{location_name}/{orig_node}-{dest_node}/{alpha_type}/{strategy}/",
    )

    # Si no existe la carpeta, crearla
    if not os.path.exists(agent_storage_path):
        os.makedirs(agent_storage_path)

    # Guardar resultados
    save_model_results(
        agent, nombre=f"QAgentSSP_", path=agent_storage_path
    )

winsound.Beep(2000, 4000)  # Beep con frecuencia de 1000 Hz durante 2 segundos
