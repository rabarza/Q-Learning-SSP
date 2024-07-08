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
from RLib.graphs.perceptron import create_perceptron_graph
from RLib.agents.ssp import QAgentSSP
from RLib.environments.ssp import SSPEnv
import numpy as np
import networkx as nx
import osmnx as ox
import winsound  # Para hacer sonar un beep al finalizar el entrenamiento
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Crear el grafo
nodes_by_layer = [1, 2, 3, 2, 1]
graph = create_perceptron_graph(nodes_by_layer, min_length=1, max_length=40)

# Definir el nodo de origen y el nodo objetivo
origin_node = ('Entrada', 0)
target_node = ('Salida', 0)
costs_distribution = "normal"

# Encontrar la política óptima, el camino más corto y la tabla Q*
policy = get_optimal_policy(graph, target_node, costs_distribution)
optimal_q_table = get_q_table_for_policy(graph, policy, target_node, costs_distribution, st=False)
shortest_path = get_shortest_path_from_policy(policy, origin_node, target_node)

# Crear el entorno SSP
environment = SSPEnv(graph, origin_node, target_node, costs_distribution, shortest_path)

# Instanciar los selectores de acción
ucb_selector = UCB1ActionSelector(c=4)
eps_selector = EpsilonGreedyActionSelector(epsilon=0.1)
exp3_selector = Exp3ActionSelector(eta='sqrt(t)')

# Instanciar los agentes
is_dynamic = True
formula = '1 / N(s,a)'
agent1 = QAgentSSP(environment=environment, dynamic_alpha=is_dynamic, alpha_formula=formula, action_selector=eps_selector)
agent2 = QAgentSSP(environment=environment, dynamic_alpha=is_dynamic, alpha_formula=formula, action_selector=ucb_selector)
agent3 = QAgentSSP(environment=environment, dynamic_alpha=is_dynamic, alpha_formula=formula, action_selector=exp3_selector)

# Realizar entrenamientos
num_episodes = 10000
agent1.train(num_episodes, shortest_path=shortest_path, q_star=optimal_q_table)
agent2.train(num_episodes, shortest_path=shortest_path, q_star=optimal_q_table)
agent3.train(num_episodes, shortest_path=shortest_path, q_star=optimal_q_table)

# Guardar los resultados
alpha_type = "dynamic" if is_dynamic else "constant"
for agent in [agent1, agent2, agent3]:
    agent_storage_path = os.path.join(
        RESULTS_DIR,
        f"Perceptron {len(nodes_by_layer)}Layers-{graph.number_of_nodes()}Nodes/{costs_distribution}/{alpha_type}_alpha/{agent.strategy}/",
    )

    # Si no existe la carpeta, crearla
    if not os.path.exists(agent_storage_path):
        os.makedirs(agent_storage_path)

    # Guardar resultados
    save_model_results(
        agent, nombre=f"QAgentSSP", path=agent_storage_path
    )
