import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))  # noqa: E402
import networkx as nx
from RLib.environments.ssp import SSPEnv
from RLib.agents.ssp import QAgentSSP
from RLib.utils.dijkstra import get_optimal_policy, get_q_table_for_policy, get_q_table_for_path, get_shortest_path_from_policy
from RLib.utils.plots import plot_results_per_episode_comp_plotly
from RLib.utils.files import save_model_results
import numpy as np
import json

from RLib.action_selection.action_selector import (
    EpsilonGreedyActionSelector,
    DynamicEpsilonGreedyActionSelector,
    UCB1ActionSelector,
    Exp3ActionSelector,
    AsOptUCBActionSelector,
)

# Definir la ubicación de los resultados
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
GRAPH_NAME = "small_graph"
# Crear un grafo simple
G = nx.DiGraph()
G.add_node(0)
G.add_node(1)
G.add_node(2)
G.add_node(3)
G.add_node(4)
G.add_edge(0, 1, length=100, speed_kph=50)
G.add_edge(1, 2, length=150, speed_kph=60)
G.add_edge(2, 4, length=200, speed_kph=40)
G.add_edge(0, 3, length=200, speed_kph=40)
G.add_edge(3, 2, length=150, speed_kph=50)
G.add_edge(3, 4, length=100, speed_kph=50)
G.add_edge(4, 2, length=100, speed_kph=50)


# Nodos de origen y destino
orig_node = 0
dest_node = 4

distribution = "lognormal"  # "uniform" o "normal"

# Calcular la política óptima y la tabla Q^*
policy = get_optimal_policy(G, dest_node, distribution=distribution)
optimal_q_table = get_q_table_for_policy(
    G, policy, dest_node, distribution=distribution)
shortest_path = shortest_path = get_shortest_path_from_policy(
    policy, orig_node, dest_node
)
optimal_q_table_for_sp = get_q_table_for_path(optimal_q_table, shortest_path)

# Serializar la tabla Q^*
json_q_star = json.dumps(optimal_q_table, indent=4)
json_q_star_for_sp = json.dumps(optimal_q_table_for_sp, indent=4)

# Guardar la tabla Q^* en un archivo
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)
with open(os.path.join(RESULTS_DIR, "optimal_q_table.json"), "w") as f:
    f.write(json_q_star)
with open(os.path.join(RESULTS_DIR, "optimal_q_table_for_sp.json"), "w") as f:
    f.write(json_q_star_for_sp)

# Configuración del entorno y del agente
env = SSPEnv(G, orig_node, dest_node, distribution, shortest_path)
agent = QAgentSSP(env, alpha_formula="0.1", action_selector=EpsilonGreedyActionSelector(epsilon=0.1))
agent_ucb = QAgentSSP(env, alpha_formula="0.1", action_selector=UCB1ActionSelector(c=.001))
agent_ao_ucb = QAgentSSP(env, alpha_formula="0.1", action_selector=AsOptUCBActionSelector())
agent_exp3 = QAgentSSP(env, alpha_formula="0.1", action_selector=Exp3ActionSelector(eta="0.3"))
NUM_EPISODES = 20000

# Entrenamiento del agente
agent.train(NUM_EPISODES, shortest_path=shortest_path, q_star=optimal_q_table)
agent_ucb.train(NUM_EPISODES, shortest_path=shortest_path, q_star=optimal_q_table)
agent_ao_ucb.train(NUM_EPISODES, shortest_path=shortest_path, q_star=optimal_q_table)
agent_exp3.train(NUM_EPISODES, shortest_path=shortest_path, q_star=optimal_q_table)

list_agents = [agent, agent_ucb, agent_ao_ucb, agent_exp3]
print("Camino más corto:", shortest_path)
print("Tabla Q^* esperada:", optimal_q_table)
print("Tabla Q aprendida por el agente:", agent.q_table)
print("Política óptima:", policy)

plot_results_per_episode_comp_plotly(list_agents, "error").show()
plot_results_per_episode_comp_plotly(list_agents, "policy error").show()
plot_results_per_episode_comp_plotly(list_agents, "average regret").show()
plot_results_per_episode_comp_plotly(list_agents, "optimal paths").show()
plot_results_per_episode_comp_plotly(list_agents, "score").show()
plot_results_per_episode_comp_plotly(list_agents, "steps").show()

for agent in list_agents:
    temp_path = f"{GRAPH_NAME}/{orig_node}-{dest_node}/dynamic_alpha/{agent.strategy}/"
    results_dir = os.path.join(RESULTS_DIR, temp_path)

    # Ruta para guardar resultados
    agent_storage_path = os.path.join(
        BASE_DIR,
        "results/",
        temp_path
    )

    # Guardar resultados
    save_model_results(
        agent, results_path=agent_storage_path
    )