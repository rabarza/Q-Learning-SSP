import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # noqa: E402
from RLib.agents.ssp import QAgentSSP
from RLib.utils.dijkstra import get_q_table_for_path
from RLib.utils.plots import plot_results_per_episode_comp_plotly
from RLib.utils.files import load_model_results, find_files_by_keyword
from RLib.utils.serializers import QAgentSSPSerializer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")


city_name = "Piedmont, California"
orig_node = 53017091
dest_node = 53035699
alpha_type = "dynamic"

ruta_carpeta = f"scripts/results/{city_name}/{orig_node}-{dest_node}/{alpha_type}_alpha/"

print(os.listdir(ruta_carpeta))
# Load QAgentSSP models
# e-greedy agents
greedy_files = find_files_by_keyword("e-", ruta_carpeta+"e-greedy")
greedy_agents = list(map(lambda x: load_model_results(
    x, ruta_carpeta+"e-greedy"), greedy_files))
# UCB1 agents
ucb_files = find_files_by_keyword("UCB1", ruta_carpeta+"UCB1")
ucb_agents = list(map(lambda x: load_model_results(
    x, ruta_carpeta+"UCB1"), ucb_files))
# exp3 agents
exp3_files = find_files_by_keyword("exp3", ruta_carpeta+"exp3")
exp3_agents = list(map(lambda x: load_model_results(
    x, ruta_carpeta+"exp3"), exp3_files))

criterias_list = ['error', 'policy error', 'score', 'steps']

for criteria in criterias_list:
    agents = greedy_agents + ucb_agents + exp3_agents
    print(agents)
    fig = plot_results_per_episode_comp_plotly(agents, criteria)
    fig.show()

# if __name__ == "__main__":
    # from RLib.utils.serializers import serialize_table
    # import json

    # for agent in agents:
    #     q_table = agent.q_table
    #     path = agent.best_path()
    #     q_table_for_sp = get_q_table_for_path(q_table, path)
    #     serialized_q_table_for_sp = serialize_table(q_table_for_sp)
    #     json_q_table_for_sp = json.dumps(serialized_q_table_for_sp, indent=4)
    #     with open(os.path.join(RESULTS_DIR, f"q_star_for_shortest_path_{city_name}_{orig_node}-{dest_node}_{agent.strategy}.json"), "w") as f:
    #         f.write(json_q_table_for_sp)
    #         f.close()
