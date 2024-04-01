from RLib.utils.file_utils import load_model_results, find_files_by_keyword
from RLib.utils.plot_utils import plot_results_per_episode_comp_plotly

# ruta_carpeta = "Santiago/results_random"
# ruta_carpeta = "Santiago/results_ucb-expectation"
# ruta_carpeta = "Santiago/results_ucb-random"
# ruta_carpeta = "Santiago/results_epsilon-expectation"
# ruta_carpeta = "Santiago/results_epsilon-random"
# ruta_carpeta = "resultados/santiago"
ruta_carpeta = "results/piedmont/"
# # # e-greedy
greedy_files = find_files_by_keyword("e-", ruta_carpeta+"e-greedy")
greedy_models = list(map(lambda x: load_model_results(x, ruta_carpeta+"e-greedy"), greedy_files))
# # # ucb1
ucb_files = find_files_by_keyword("UCB1", ruta_carpeta+"UCB1")
ucb_models = list(map(lambda x: load_model_results(x, ruta_carpeta+"UCB1"), ucb_files))
# # # # exp3
exp3_files = find_files_by_keyword("exp3", ruta_carpeta+"exp3")
exp3_models = list(map(lambda x: load_model_results(x, ruta_carpeta+"exp3"), exp3_files))

criterias = ['steps', 'score', 'error'] 
# criterias = ['error'] 
for criteria in criterias:
    models = greedy_models + ucb_models + exp3_models
    plot_results_per_episode_comp_plotly(models, criteria=criteria, compare_best=False)
