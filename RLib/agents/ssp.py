import os  # noqa: E402
import sys  # noqa: E402

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))  # noqa: E402

from typing import Tuple, Union
from RLib.environments.ssp import SSPEnv
from RLib.utils.tables import (
    max_q_table,
    max_norm,
    dict_states_actions_zeros,
    dict_states_zeros,
)
from RLib.action_selectors import ActionSelector, EpsilonGreedyActionSelector
from stqdm import stqdm
from tqdm import tqdm
from math import sqrt, log  # util para el calculo de la tasa de aprendizaje en eval
from typing import Dict, Any
import numpy as np
import random


class QAgent:
    """
    Clase base para el agente Q-Learning.
    """

    def __init__(
        self,
        action_selector: ActionSelector = EpsilonGreedyActionSelector(0.1),
        alpha: Union[str, float] = 0.1,
        gamma: float = 1,
    ):
        """
        Parameters
        ----------
        alpha: Union[str, float]
            fórmula para calcular el valor de alpha. Puede ser cualquier expresión matemática válida que contenga las variables 'N(s,a)', 'N(s)', 't', 'sqrt' y 'log'.
            ej: '1 / N(s,a)', '1000 / (N(s) + 1000)', '1 / sqrt(N(s,a))', '1 / log(N(s,a) + 1)', etc.

        gamma: float
            factor de descuento. Debe ser un valor entre 0 y 1.

        action_selector: ActionSelector (objeto de la clase ActionSelector)
            selector de acciones.
        """
        self.alpha = alpha
        self.gamma = gamma
        self.action_selector = action_selector
        self.strategy = self.action_selector.strategy
        self.env = None
        self.q_table = dict()
        self.visits_actions = dict()
        self.visits_states = dict()
        self.actual_episode = 0
        # Inicializar dynamic_alpha y alpha_type después de establecer alpha
        self.dynamic_alpha = self.is_dynamic_alpha(self.alpha)
        self.alpha_type = "dynamic" if self.dynamic_alpha else "constant"
        self.alpha_eval_expr = (
            str(alpha).replace("N(s,a)", "N_sa").replace("N(s)", "N_s")
        )
        # donde se guardan los resultados del entrenamiento
        self.storage_path = None

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(strategy={self.action_selector} alpha_type={self.alpha_type}, discount={self.gamma})"  # noqa: E501

    def __repr__(self) -> str:
        return self.__str__()

    @staticmethod
    def is_dynamic_alpha(alpha):
        try:
            float(alpha)
            return False
        except ValueError or TypeError:
            return True

    def eval_alpha(self, state, action):
        """
        Retorna el valor de alpha para el estado y acción indicados en el tiempo de actualización.
        """
        context = {
            "N_sa": self.visits_actions[state][action] + 1,
            "N_s": self.visits_states[state] + 1,
            "t": self.actual_episode + 1,
            "sqrt": sqrt,
            "log": log,
        }
        return eval(self.alpha_eval_expr, context)

    def argmax_q_table(self, state):
        """
        Retorna la acción a con mayor valor Q(s,a) para un estado s
        """
        available_actions = self.action_set(state)
        q_values = {action: self.q_table[state][action] for action in available_actions}
        argmax_action = max(q_values, key=q_values.get)
        return argmax_action

    def max_q_table(self, state):
        """
        Retorna el valor máximo Q(s,a) para un estado s
        """
        assert state in self.q_table, f"El estado {state} no está en q_table."
        assert self.q_table[state], f"No hay acciones disponibles en el estado {state}"
        return max(list(self.q_table[state].values()))

    def calculate_max_norm_errors(
        self, q_table: dict, q_star: dict, state_action_pairs: list
    ) -> Tuple[float, float]:
        if q_star and state_action_pairs:
            max_norm_error = max_norm(q_table, q_star)
            max_norm_error_shortest_path = max_norm(
                q_table, q_star, path=state_action_pairs
            )
        elif q_star:
            max_norm_error = max_norm(q_table, q_star)
            max_norm_error_shortest_path = 0
        else:
            max_norm_error = 0
            max_norm_error_shortest_path = 0

        return max_norm_error, max_norm_error_shortest_path

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
        self.visits_states[state] += 1

    def increment_times_state_action(self, state, action):
        """
        Incrementa la cantidad de veces que se toma una acción en un estado
        """
        self.visits_actions[state][action] += 1

    def select_action(self, state: Any, increase_visits: bool = True) -> Any:
        """
        Seleccionar la siguiente acción a tomar

        Parámetros:
        -----------
            state (int): Estado actual
            increase_visits (bool): Indica si se debe incrementar la cantidad de veces que se toma la acción

        Retorna:
        --------
            action (int): Acción a tomar
        """
        # Seleccionar la siguiente acción a tomar
        action = self.action_selector.select_action(self, state)
        # Incrementar la cantidad de veces que se toma la acción
        increase_visits and self.increment_times_state_action(state, action)
        # Devolver acción
        return action

    def initialize_results(self):
        """Inicializa los contenedores para almacenar los resultados del entrenamiento."""
        self.steps = []
        self.scores = []
        self.avg_scores = []
        self.regret = []
        self.average_regret = []

    def update_steps_and_scores(self, steps: int, score: float):
        """Actualiza los pasos y los scores obtenidos en cada episodio."""
        self.steps.append(steps)
        self.scores.append(score)
        self.avg_scores.append(np.mean(self.scores))

    def update_regret(self, episode: int, optimal_cost: float):
        """Actualiza el regret acumulado y promedio."""
        self.regret.append(episode * optimal_cost - sum(self.scores[: episode + 1]))
        self.average_regret.append(
            optimal_cost - sum(self.scores[: episode + 1]) / max(episode, 1)
        )

    def update_results(
        self, episode: int, steps: int, score: float, optimal_cost: float
    ):
        self.update_steps_and_scores(steps, score)
        self.update_regret(episode, optimal_cost)

    def update_q_table(
        self, state: Any, action: Any, reward: float, next_state: Any
    ) -> None:
        """
        Actualiza el valor de Q(s,a) en la tabla Q usando Q-Learning
        """
        alpha = self.eval_alpha(state, action)
        gamma = self.gamma
        q_old = self.q_table[state][action]
        q_new = q_old * (1 - alpha) + alpha * (
            reward + gamma * self.max_q_table(next_state)
        )
        self.q_table[state][action] = q_new

    def save(self, path: str):
        """
        Guardar el agente en un archivo .pkl y los resultados en un archivo .json
        """
        from RLib.utils.files import save_model_results

        save_path = os.path.join(
            path, f"{self.alpha_type}_alpha/{self.strategy}/"
        )
        self.storage_path = save_path
        save_model_results(self, save_path)

    def results(self):
        # Devuelve los resultados como diccionario
        optimal_cost = max_q_table(self.q_star, self.env.start_state)
        results = {
            "strategy": self.strategy,
            "parameters": (
                self.action_selector.get_label()
                if self.action_selector
                else self.get_label()
            ),
            "steps": self.steps,
            "scores": self.scores,
            "avg_scores": self.avg_scores,
            "regret": self.regret,
            "average_regret": self.average_regret,
            "max_norm_error": self.max_norm_error,
            "max_norm_error_shortest_path": self.max_norm_error_shortest_path,
            "max_norm_error_normalized": np.array(self.max_norm_error)
            / abs(optimal_cost),
            "max_norm_error_shortest_path_normalized": np.array(
                self.max_norm_error_shortest_path
            )
            / abs(optimal_cost),  # noqa: E501
            "optimal_cost": optimal_cost,
            "optimal_paths": self.optimal_paths,
            "alpha": self.alpha,
        }
        return results


class QAgentSSP(QAgent):
    """
    Agente que resuelve el Stochastic Shortest Path Problem (SSP) mediante Q-Learning.
    """

    def __init__(
        self,
        environment: SSPEnv,
        action_selector: ActionSelector = EpsilonGreedyActionSelector(0.1),
        alpha: Union[str, float] = 0.1,
        gamma: float = 1,
    ):
        """
        Parameters
        ----------

        environment: SSPEnv
            entorno en el que se encuentra el agente (objeto de la clase SSPEnv).

        alpha: Union[str, float]
            fórmula para calcular el valor de alpha. 
            Puede ser cualquier expresión matemática válida que contenga las variables `N(s,a)`: visitas al estado `s` y acción `a`, `N(s)`: visitas al estado `s`, y `t`: episodio. Se pueden usar las funciones `sqrt` y `log` e.g.::
        
            '1 / N(s)'
            '1 / N(s,a)'
            '1000 / (N(s,a) + 1000)'
            '1 / sqrt(t)'
            '1 / log(N(s,a) + 1)'
            
        gamma: float
            factor de descuento. Debe ser un valor entre 0 y 1. Por defecto es 1 (sin descuento).

        action_selector: ActionSelector
            selector de acciones (objeto de la clase ActionSelector). Por defecto es EpsilonGreedyActionSelector(0.1).
        
        """
        super().__init__(alpha=alpha, gamma=gamma, action_selector=action_selector)
        self.env = environment
        self.num_states = environment.num_states
        self.num_actions = environment.num_actions
        # Se cuenta la cantidad de veces que se tomo una accion en cada estado N(s,a)
        self.visits_actions = dict_states_actions_zeros(self.env.graph)
        # Se cuenta la cantidad de veces que se visita un estado N(s)
        self.visits_states = dict_states_zeros(self.env.graph)
        # Se inicializa la matriz Q(s,a) con valores aleatorios
        self.q_table = dict_states_actions_zeros(self.env.graph)
        # Inicializar los contenedores de resultados
        self.initialize_results()

        self.id = id(self)

    def initialize_results(self):
        """Inicializa los contenedores para almacenar los resultados del entrenamiento."""
        super().initialize_results()
        self.optimal_paths = []
        self.max_norm_error = []
        self.max_norm_error_shortest_path = []

    def update_results(
        self,
        episode,
        steps,
        score,
        optimal_cost,
        max_norm_error,
        max_norm_error_shortest_path,
        optimal_paths,
    ):
        """Actualiza los resultados después de cada episodio."""
        super().update_results(episode, steps, score, optimal_cost)
        self.max_norm_error.append(max_norm_error)
        self.max_norm_error_shortest_path.append(
            max_norm_error_shortest_path
        )  # noqa: E501
        self.optimal_paths.append(optimal_paths)

    def train(
        self,
        num_episodes: int = 100,
        shortest_path: list = None,
        q_star: Dict[Any, Dict[Any, float]] = None,
        verbose: bool = False,
        use_streamlit: bool = False,
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
            Indica si se debe mostrar información de la ejecución de cada episodio. The default is False.

        Returns
        -------
        None
        """

        self.num_episodes = num_episodes
        self.shortest_path = shortest_path

        # optimal q table (Q*)
        self.q_star = q_star

        # Estado inicial
        initial_state = self.env.start_state
        # optimal cost (used to calculate regret)
        optimal_cost = max_q_table(q_star, initial_state)
        optimal_paths_count = 0

        # Comenzar a entrenar al agente
        progress_bar = tqdm if not use_streamlit else stqdm
        episodes_range = progress_bar(
            range(num_episodes), desc="Completado", ncols=100, leave=True
        )  # noqa: E501
        for episode in episodes_range:
            self.env.reset()
            done = False
            self.actual_episode = episode
            total_score = 0
            path = []
            steps = 0
            state = initial_state
            while not done:
                path.append(state)
                # Se realiza la transición (action, next_state, reward)
                action = self.select_action(state)
                next_state, reward, done, info = self.env.take_action(state, action)
                # Actualizar valores Q_table
                self.update_q_table(state, action, reward, next_state)
                # Incrementar cantidad de visitas al estado
                self.increment_times_state(state)
                # Ir al estado siguiente
                state = next_state
                # Aumentar la cantidad de pasos del episodio
                steps += 1
                # Acumular el puntaje obtenido en el episodio
                total_score += reward
                # Imprimir información de la ejecución
                print(info) if verbose else None

            if path == self.env.shortest_path:
                optimal_paths_count += 1
            # Calcular el error de la norma máxima entre la tabla Q y la tabla Q*
            max_norm_error, max_norm_error_shortest_path = (
                self.calculate_max_norm_errors(self.q_table, q_star, shortest_path)
            )

            self.update_results(
                episode,
                steps,
                total_score,
                optimal_cost,
                max_norm_error,
                max_norm_error_shortest_path,
                optimal_paths_count,
            )
            normalized_error = max_norm_error / -optimal_cost
            normalized_error_shortest_path = (
                max_norm_error_shortest_path / -optimal_cost
            )
            # Mostrar información de la ejecución
            message = f"Episodio {episode}/{num_episodes} - Pasos: {self.steps[episode]} - Max norm error: {normalized_error:.3f} - Max norm error path: {normalized_error_shortest_path:.3f}"
            # Mostrar mensaje en streamlit o en consola
            episodes_range.set_description(
                "Episodio {}/{}".format(episode, num_episodes)
            )
            # Mostrar mensaje en intervalos específicos
            if episode % 100 == 0 or episode == num_episodes - 1:
                progress_bar.write(message)

    def best_path(self, state: Any = None) -> list:
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
        # Devuelve los resultados como diccionario
        optimal_cost = max_q_table(self.q_star, self.env.start_state)
        results = {
            "strategy": self.strategy,
            "alpha": self.alpha,
            "parameters": (
                self.action_selector.get_label()
                if self.action_selector
                else self.get_label()
            ),
            "steps": self.steps,
            "scores": self.scores,
            "avg_scores": self.avg_scores,
            "regret": self.regret,
            "average_regret": self.average_regret,
            "max_norm_error": self.max_norm_error,
            "max_norm_error_shortest_path": self.max_norm_error_shortest_path,
            "max_norm_error_normalized": np.array(self.max_norm_error)
            / abs(optimal_cost),
            "max_norm_error_shortest_path_normalized": np.array(
                self.max_norm_error_shortest_path
            )
            / abs(optimal_cost),  # noqa: E501
            "optimal_cost": optimal_cost,
            "optimal_paths": self.optimal_paths,
        }
        return results


class QAgentSSPSarsa0(QAgentSSP):
    """
    Agente que resuelve el Stochastic Shortest Path Problem (SSP) mediante SARSA(0).
    """

    def update_q_table(self, state, action, reward, next_state):
        """
        Actualiza el valor de Q(s,a) en la tabla Q usando SARSA(0)
        """
        alpha = self.eval_alpha(state, action)
        gamma = self.gamma
        next_action = self.select_action(next_state, increase_visits=False)
        q_old = self.q_table[state][action]
        next_q = self.q_table[next_state][next_action]
        q_new = q_old * (1 - alpha) + alpha * (reward + gamma * next_q)
        self.q_table[state][action] = q_new


class QAgentSSPExpectedSarsa0(QAgentSSPSarsa0):
    """
    Agente que resuelve el Stochastic Shortest Path Problem (SSP) mediante Expected SARSA(0).
    """

    def update_q_table(self, state, action, reward, next_state):
        """
        Actualiza el valor de Q(s,a) en la tabla Q usando Expected-SARSA(0)
        """
        alpha = self.eval_alpha(state, action)
        gamma = self.gamma
        q_old = self.q_table[state][action]
        expected_next_q = self.expected_q_value(next_state)
        q_new = q_old * (1 - alpha) + alpha * (reward + gamma * expected_next_q)
        self.q_table[state][action] = q_new

    def expected_q_value(self, state):
        """
        Calcula el valor esperado de Q(s,a) para un estado s
        """
        available_actions = self.action_set(state)
        action_probabilities = self.action_selector.get_probabilities(self, state)
        q_values = self.q_table[state]
        expected_q = np.sum(
            [
                action_probabilities[action] * q_values[action]
                for action in available_actions
            ]
        )
        return expected_q


if __name__ == "__main__":
    from RLib.environments.ssp import SSPEnv
    from RLib.graphs.perceptron import create_perceptron_graph, plot_network_graph
    from RLib.utils.dijkstra import (
        get_shortest_path_from_policy,
        get_optimal_policy_and_q_star,
    )
    from RLib.action_selectors import EpsilonGreedyDecayActionSelector

    # Crear el grafo y el entorno
    graph = create_perceptron_graph([1, 10, 1], 20, 2000, 20)

    start_node = 1
    end_node = 0
    speed_distribution = "uniform"

    environment = SSPEnv(
        graph, start_node, end_node, costs_distribution=speed_distribution
    )

    # Obtener la política óptima y la tabla Q para la política óptima
    optimal_policy, optimal_q_table = get_optimal_policy_and_q_star(
        graph, end_node, speed_distribution
    )

    # Obtener lista de nodos del camino más corto
    shortest_path = get_shortest_path_from_policy(optimal_policy, start_node, end_node)

    # Crear el selector de acciones y el agente Q-Learning
    eps_selector = EpsilonGreedyActionSelector(epsilon=1)
    alpha_expr = "1000 / (1000 + N(s,a))"
    # Entrenar agente Q-Learning
    q_agent = QAgentSSP(
        environment=environment, alpha=alpha_expr, action_selector=eps_selector
    )
    q_agent.train(
        num_episodes=20000, shortest_path=shortest_path, q_star=optimal_q_table
    )

    # Entrenar agente SARSA(0)
    eps_decay_selector = EpsilonGreedyDecayActionSelector(constant=1)
    sarsa_agent = QAgentSSPSarsa0(
        environment=environment, alpha=alpha_expr, action_selector=eps_decay_selector
    )
    sarsa_agent.train(
        num_episodes=30000, shortest_path=shortest_path, q_star=optimal_q_table
    )
    # Entrenar agente Expected SARSA(0)
    expected_sarsa_agent = QAgentSSPExpectedSarsa0(
        environment=environment, alpha=alpha_expr, action_selector=eps_decay_selector
    )
    expected_sarsa_agent.train(
        num_episodes=30000, shortest_path=shortest_path, q_star=optimal_q_table
    )
    # Guardar los resultados
    q_agent.save("results")
    sarsa_agent.save("results")
