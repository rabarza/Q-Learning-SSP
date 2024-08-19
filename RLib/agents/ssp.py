import matplotlib.pyplot as plt
from RLib.environments.ssp import SSPEnv
from RLib.utils.tables import max_q_table, max_norm
from RLib.utils.files import save_model_results
from RLib.action_selectors import EpsilonGreedyActionSelector
from stqdm import stqdm
from tqdm import tqdm
from math import sqrt, log  # util para el calculo de la tasa de aprendizaje en eval
import numpy as np
import random
import copy
import sys
import os
# Importar RLib desde el directorio superior
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))


class QAgent:
    """
    Clase base para el agente Q-Learning.
    """

    def __init__(self):
        # ruta de almacenamiento de los resultados
        self.storage_path = None

    def argmax_q_table(self, state):
        """
        Retorna la acción a con mayor valor Q(s,a) para un estado s
        """
        available_actions = self.action_set(state)
        q_values = {action: self.q_table[state][action]
                    for action in available_actions}
        argmax_action = max(q_values, key=q_values.get)
        return argmax_action

    def max_q_table(self, state):
        """
        Retorna el valor máximo Q(s,a) para un estado s
        """
        assert state in self.q_table, f"El estado {state} no está en q_table."
        assert self.q_table[state], f"No hay acciones disponibles en el estado {state}"
        return max(list(self.q_table[state].values()))

    def action_set(self, state) -> list:
        """
        Retorna el conjunto de acciones disponibles en el estado s

        Parámetros:
        ----------
        state (int): Estado actual

        Retorna:
        --------
        available_actions (list): Lista de acciones disponibles en el estado s
        """
        # Acciones disponibles en el estado s
        return self.env.action_set(state)

    def random_action(self, state):
        """
        Retorna una acción aleatoria a' de Q(s,a')
        """
        assert state in self.q_table, f"El estado {state} no está en q_table."
        assert self.q_table[state], f"No hay acciones disponibles en el estado {state}"
        actions = self.action_set(state)
        return random.choice(actions)

    def number_of_available_actions(self, state):
        """
        Retorna la cantidad de acciones disponibles en el estado s
        """
        return len(self.action_set(state))

    def increment_times_state(self, state):
        """
        Incrementa la cantidad de veces que se visita un estado
        """
        self.times_states[state] += 1

    def increment_times_state_action(self, state, action):
        """
        Incrementa la cantidad de veces que se toma una acción en un estado
        """
        self.times_actions[state][action] += 1

    def get_alpha(self, state, action):
        """
        Retorna el valor de alpha para el estado y acción indicados.
        """
        alpha = self.alpha
        if not self.dynamic_alpha:
            return alpha
        # Reemplazar N(s,a) por t en la fórmula de alpha
        if "N(s,a)" in self.alpha_formula:
            # Realizar conteo de visitas al par estado-acción
            t = self.times_actions[state][action]
            formula = self.alpha_formula.replace("N(s,a)", "t")
        elif "N(s)" in self.alpha_formula:
            # Realizar conteo de visitas al estado
            t = max(self.times_states[state], 1)
            formula = self.alpha_formula.replace("N(s)", "t")
        elif "t" in self.alpha_formula:
            t = self.actual_episode + 1
            formula = self.alpha_formula
        else:
            formula = self.alpha_formula
        # Evaluar la fórmula de alpha dinámico con el valor de t
        alpha_value = eval(formula)
        return alpha_value

    def select_action(self, state):
        """
        Seleccionar la siguiente acción a tomar
        Parámetros:
            state (int): Estado actual
        Retorna:
            action (int): Acción a tomar
        """
        # Seleccionar la siguiente acción a tomar
        action = self.action_selector.select_action(self, state)
        # Incrementar el contador de visitas para la acción en el estado actual
        self.increment_times_state_action(state, action)
        # Devolver acción
        return action

    def save(self, path):
        """
        Guardar el agente en un archivo .pkl y los resultados en un archivo .json
        """
        alpha_type = "dynamic" if self.dynamic_alpha else "constant"
        save_path = os.path.join(path, f"{alpha_type}_alpha/{self.strategy}/")  # noqa: E501
        save_model_results(self, save_path)


class QAgentSSP(QAgent):
    """
    Agente que resuelve el problema del laberinto usando el algoritmo Q-Learning.
    """

    def __init__(
        self,
        environment: SSPEnv,
        alpha=0.01,
        gamma=1,
        dynamic_alpha=False,
        alpha_formula="alpha",
        action_selector=EpsilonGreedyActionSelector(epsilon=0.1),
    ):
        """
        Parámetros:

        `environment`: SSPEnv (objeto de la clase SSPEnv)
            entorno en el que se encuentra el agente.

        `epsilon`: float
            probabilidad de exploración. Se utiliza en las estrategias e-greedy, e-truncated y e-decay.

        `alpha`: float
            tasa de aprendizaje. Se utiliza en el algoritmo Q-Learning. Debe ser un valor entre 0 y 1.

        `gamma`: float
            factor de descuento. Se utiliza en el algoritmo Q-Learning. Debe ser un valor entre 0 y 1.

        `dynamic_alpha`: bool
            indica si se debe utilizar alpha dinámico.

        `alpha_formula`: str
            fórmula para calcular el valor de alpha. Puede ser: 'max(alpha, 1 / N(s,a))', '1 / N(s,a)' o 'alpha'. Por defecto es 'alpha'.

        `action_selector`: ActionSelector (objeto de la clase ActionSelector)
            selector de acciones.

        """
        super().__init__()
        self.env = environment
        self.num_states = environment.num_states
        self.num_actions = environment.num_actions
        self.alpha = alpha
        self.dynamic_alpha = True if dynamic_alpha or alpha_formula != "alpha" else False
        self.alpha_formula = alpha_formula
        self.gamma = gamma
        self.action_selector = action_selector
        self.strategy = action_selector.strategy
        # Se cuenta la cantidad de veces que se tomo una accion en cada estado N(s,a)
        self.times_actions = self.env.dict_states_actions_zeros()
        # Se cuenta la cantidad de veces que se visita un estado N(s)
        self.times_states = self.env.dict_states_zeros()
        # Se inicializa la matriz Q(s,a) con valores aleatorios
        self.q_table = self.env.dict_states_actions_zeros()
        # self.q_table = self.env.dict_states_actions_constant(constant=-15)
        self.id = id(self)

    def __str__(self) -> str:
        return f"QAgentSSP(strategy={self.action_selector} alpha={self.alpha} gamma={self.gamma} alpha_formula={self.alpha_formula})"

    def __repr__(self) -> str:
        return self.__str__()

    def train(
        self,
        num_episodes=100,
        shortest_path=None,
        q_star=None,
        verbose=False,
        use_streamlit=False
    ) -> None:
        """
        Resuelve el problema del Shortest Path usando el algoritmo Q-Learning

        Parameters
        ----------
        num_episodes : int
            Número de episodios a ejecutar. The default is 100.

        shortest_path : dict, optional
            Camino más corto entre el estado inicial y el estado terminal. The default is None.

        q_star : dict, optional
            Tabla Q* óptima. The default is None.

        verbose : bool, optional
            Indica si se debe mostrar información de la ejecución. The default is True.

        Returns
        -------
        None
        """

        self.num_episodes = num_episodes
        self.shortest_path = shortest_path

        self.steps = np.zeros(num_episodes)
        self.scores = np.zeros(num_episodes)
        self.avg_scores = np.zeros(num_episodes)
        self.regret = np.zeros(num_episodes)
        self.average_regret = np.zeros(num_episodes)
        self.optimal_paths = np.zeros(num_episodes)
        optimal_paths_count = 0
        # optimal q table (Q*)
        self.q_star = q_star
        # max norm error for all state-action pairs
        self.max_norm_error = np.zeros(num_episodes)
        # max_norm_error for shortest path state-action pairs only
        self.max_norm_error_shortest_path = np.zeros(num_episodes)
        # discount rate
        gamma = self.gamma
        # Estado inicial
        initial_state = self.env.start_state
        # optimal cost (used to calculate regret)
        optimal_cost = max_q_table(q_star, initial_state)

        # Comenzar a entrenar al agente
        progress_bar = tqdm if not use_streamlit else stqdm
        episodes_range = progress_bar(
            range(num_episodes), desc="Completado", ncols=100, leave=True)
        for episode in episodes_range:
            self.env.reset()
            done = False
            self.actual_episode = episode
            total_score = 0
            path = []
            state = initial_state
            while not done:
                path.append(state)
                # Se realiza la transición (action, next_state, reward)
                action = self.select_action(state)
                alpha = self.get_alpha(state, action)
                next_state, reward, done, info = self.env.take_action(
                    state, action
                )

                # Actualizar valores Q_table
                q_old = self.q_table[state][action]
                q_new = q_old * (1 - alpha) + alpha * (
                    reward + gamma * self.max_q_table(next_state)
                )
                # print(f"q_old: {q_old}, q_new: {q_new}, alpha: {alpha}, reward: {reward}, visits: {self.times_actions[state][action]}")
                # Almacenar el nuevo valor de Q(s,a)
                self.q_table[state][action] = q_new
                # Incrementar cantidad de visitas al estado
                self.increment_times_state(state)
                # Ir al estado siguiente
                state = next_state
                # Aumentar la cantidad de pasos del episodio
                self.steps[episode] += 1
                # Acumular el puntaje obtenido en el episodio
                total_score += reward
                # Imprimir información de la ejecución
                print(info) if verbose else None

            path.append(state)  # Agregar el estado terminal al camino
            # Contar la cantidad de veces que se llegó al camino óptimo
            if path == self.env.shortest_path:
                optimal_paths_count += 1
            # Calcular el error de la norma máxima entre la tabla Q y la tabla Q*
            if q_star and shortest_path:
                max_norm_error = max_norm(self.q_table, q_star)
                max_norm_error_shortest_path = max_norm(
                    self.q_table, q_star, path=shortest_path
                )
            elif q_star:
                max_norm_error = max_norm(self.q_table, q_star)
                max_norm_error_shortest_path = 0
            else:
                max_norm_error = 0
                max_norm_error_shortest_path = 0

            # Almacenar el error de la norma máxima
            self.max_norm_error[episode] = max_norm_error
            self.max_norm_error_shortest_path[episode] = max_norm_error_shortest_path
            # Almacenar cantidad promedio de valores q y del episodio
            self.scores[episode] = total_score
            self.avg_scores[episode] = np.sum(self.scores[:episode+1])/max(episode, 1)  # noqa: E501
            # Calcular el regret
            self.average_regret[episode] = optimal_cost - np.sum(self.scores[:episode+1])/max(episode, 1)  # noqa: E501
            self.regret[episode] = episode*optimal_cost - np.sum(self.scores[:episode+1])  # noqa: E501
            self.optimal_paths[episode] = optimal_paths_count

            # Mostrar información de la ejecución
            message = f"Episodio {episode + 1}/{num_episodes} - Puntaje: {total_score:.2f} - Pasos: {self.steps[episode]} - Max norm error: {max_norm_error:.3f} - Max norm error path: {max_norm_error_shortest_path:.3f}\n"
            # Mostrar mensaje en streamlit o en consola
            progress_bar.write(message)

    def best_path(self, state=None) -> list:
        """Devuelve el mejor camino desde un estado inicial hasta el estado terminal

        Parámetros:
        -----------
        state: int
            estado inicial desde el cual se quiere encontrar el mejor camino hasta el estado terminal

        """
        if not state:
            state = self.env.start_state
        path = []
        if self.env.terminal_state == state:
            return path
        path.append(state)
        done = False
        while not (done):
            action = self.argmax_q_table(state)
            state = self.env.take_action(state, action)[0]
            path.append(state)
            if self.env.terminal_state == state:
                done = True
        return path
    
    def results(self):
        # make a dictionary with the results
        results = {
            "strategy": self.strategy,
            "parameters": self.action_selector.get_label(),
            "steps": self.steps,
            "scores": self.scores,
            "avg_scores": self.avg_scores,
            "regret": self.regret,
            "average_regret": self.average_regret,
            "max_norm_error": self.max_norm_error,
            "max_norm_error_shortest_path": self.max_norm_error_shortest_path
        }
        return results


if __name__ == "__main__":
    from RLib.environments.ssp import SSPEnv
    from RLib.graphs.perceptron import create_perceptron_graph, plot_network_graph
    from RLib.utils.dijkstra import get_optimal_policy, get_q_table_for_policy, get_shortest_path_from_policy

    # Crear el grafo y el entorno
    graph = create_perceptron_graph([1, 20, 1], 100, 2000)
    start_node = ('Entrada', 0)
    end_node = ('Salida', 0)
    environment = SSPEnv(graph, start_node, end_node,
                         costs_distribution="lognormal")
    # Obtener la política óptima y la tabla Q para la política óptima
    policy = get_optimal_policy(environment.graph, end_node)
    optimal_q_table = get_q_table_for_policy(
        environment.graph, policy, end_node, st=False)
    # Obtener lista de nodos del camino más corto
    shortest_path = get_shortest_path_from_policy(policy, start_node, end_node)
    # Crear el selector de acciones y el agente Q-Learning
    eps_selector = EpsilonGreedyActionSelector(epsilon=0.1)
    agent = QAgentSSP(environment=environment, dynamic_alpha=True,
                      alpha_formula='1 / N(s,a)', action_selector=eps_selector)
    # Entrenar al agente
    agent.train(num_episodes=10000, distribution='lognormal',
                shortest_path=shortest_path, q_star=optimal_q_table)
    print(agent.best_path())

