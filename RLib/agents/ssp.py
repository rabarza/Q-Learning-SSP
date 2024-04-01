import numpy as np
import matplotlib.pyplot as plt
import random
import copy
from tqdm import tqdm
from datetime import datetime
from RLib.utils.table_utils import max_norm, exploitation
from stqdm import stqdm
from RLib.action_selection.action_selector import EpsilonGreedyActionSelector


class QAgentSSP:
    """
    Agente que resuelve el problema del laberinto usando el algoritmo Q-Learning.
    """

    def __init__(
        self,
        environment,
        alpha=0.01,
        gamma=1,
        dynamic_alpha=False,
        action_selector=EpsilonGreedyActionSelector(epsilon=0.1),
    ):
        """
        Parámetros:

        `environment` -- entorno en el que se encuentra el agente.

        `epsilon` -- probabilidad de exploración. Se utiliza en las estrategias e-greedy, e-truncated y e-decay.

        `alpha` -- tasa de aprendizaje. Se utiliza en el algoritmo Q-Learning. Debe ser un valor entre 0 y 1.

        `gamma` -- factor de descuento. Se utiliza en el algoritmo Q-Learning. Debe ser un valor entre 0 y 1.

        `dynamic_alpha` -- indica si se debe utilizar alpha dinámico.

        `action_selector` -- selector de acciones.

        """

        self.env = environment
        self.num_states = environment.num_states
        self.num_actions = environment.num_actions

        self.alpha = alpha
        self.gamma = gamma
        self.action_selector = action_selector
        self.strategy = action_selector.strategy

        # Se cuenta la cantidad de veces que se tomo una accion en cada estado N(s,a)
        self.times_actions = self.env.dict_states_actions_zeros()

        # Se cuenta la cantidad de veces que se visita un estado N(s)
        self.times_states = self.env.dict_states_zeros()

        # Se inicializa la matriz Q(s,a) con valores aleatorios
        self.q_table = self.env.dict_states_actions_zeros()

        # self.id = datetime.now().strftime("%Y%m%d%H%M%S")

        # alpha dinámico
        if dynamic_alpha:
            self.alpha_formula = "max(alpha, 1 / N(s,a))"
        else:
            self.alpha_formula = "alpha"

    def __str__(self):
        return f"QAgentSSP(strategy={self.strategy}, epsilon={self.action_selector.epsilon}, alpha={self.alpha}, gamma={self.gamma})\n"

    def argmax_q_table(self, state):
        """
        Retorna la acción a con mayor valor Q(s,a) para un estado s
        """
        argmax_action = max(self.q_table[state], key=self.q_table[state].get)
        return argmax_action

    def max_q_table(self, state):
        """
        Retorna el valor máximo Q(s,a) para un estado s
        """
        if state in self.q_table and self.q_table[state]:
            return max(list(self.q_table[state].values()))
        else:
            # Devuelve un valor predeterminado (por ejemplo, 0) si el estado no está en q_table o su diccionario de acciones está vacío.
            raise Exception(
                f"El estado {state} no está en q_table o su diccionario de acciones está vacío."
            )

    def random_action(self, state):
        """
        Retorna una acción aleatoria a' de Q(s,a')
        """
        keys = list(self.q_table[state].keys())
        size = len(keys)
        index = np.random.randint(0, size)
        return keys[index]

    def number_of_available_actions(self, state):
        """
        Retorna la cantidad de acciones disponibles en el estado s
        """
        return len(self.q_table[state])

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
        if self.alpha_formula == "max(alpha, 1 / N(s,a))":
            return max(self.alpha, 1 / (self.times_actions[state][action] + 1))
        else:
            return self.alpha

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

    def train(
        self,
        num_episodes=100,
        distribution="expectation-lognormal",
        policy=None,
        q_star=None,
        verbose=False,
    ):
        """
        Resuelve el problema del Shortest Path usando el algoritmo Q-Learning

        Parameters
        ----------
        num_episodes : int
            Número de episodios a ejecutar. The default is 100.
        distribution : str, optional
            Distribución de probabilidad que se utiliza para generar los valores de recompensa. Puede ser:
                - 'expectation-lognormal': distribución lognormal con media igual a la recompensa esperada.
                - 'lognormal': distribución lognormal con media igual a 1.
        policy : dict, optional
            Política óptima definida sólo para los nodos del camino más corto, notar que no está definida para todos los nodos del grafo. The default is None.
        q_star : dict, optional
            Tabla Q* óptima. The default is None.
        verbose : bool, optional
            Indica si se debe mostrar información de la ejecución. The default is True.

        Returns
        -------
        None
        """
        from RLib.utils.table_utils import max_norm, exploitation

        self.num_episodes = num_episodes
        self.policy = policy
        self.distribution = distribution

        self.steps = np.zeros(num_episodes)
        self.scores = np.zeros(num_episodes)
        self.avg_scores = np.zeros(num_episodes)
        
        self.q_star = q_star

        # best
        self.steps_best = np.zeros(num_episodes)  # Cambios temporales
        self.scores_best = np.zeros(num_episodes)  # Cambios temporales
        self.avg_scores_best = np.zeros(num_episodes)  # Cambios temporales

        # max norm error
        self.max_norm_error = np.zeros(num_episodes)
        # max_norm_error for a policy
        self.max_norm_error_policy = np.zeros(num_episodes)

        q_table_aux = copy.deepcopy(self.q_table)  # Cambios temporales

        gamma = self.gamma
        # inicializar tabla de valores q
        # self.q_table = self.env.dict_states_actions_random()

        for episode in stqdm(
            range(num_episodes), desc="Completado", ncols=100, leave=True
        ):
            done = False
            self.actual_episode = episode
            total_score = 0

            state = self.env.start_state
            while not done:
                # Se realiza la transición (action, next_state, reward)
                action = self.select_action(state)
                alpha = self.get_alpha(state, action)

                next_state, reward, done, info = self.env.take_action(
                    state, action, distribution
                )
                # Actualizar valores Q_table
                q_old = self.q_table[state][action]
                q_new = q_old * (1 - alpha) + alpha * (
                    reward + gamma * self.max_q_table(next_state)
                )
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

            # Calcular el error de la norma máxima entre la tabla Q y la tabla Q*
            if not q_star:
                q_table_aux, score_aux, steps_aux = exploitation(
                    self.q_table,
                    self.steps[episode],
                    self.env,
                    tolerance=2,
                    alpha=alpha,
                    gamma=gamma,
                )
                self.steps_best[episode] = steps_aux
                self.scores_best[episode] = score_aux
                self.avg_scores_best[episode] = score_aux / max(
                    self.steps_best[episode], 1
                )
                max_norm_error_policy = max_norm(
                    self.q_table, q_table_aux, policy=policy
                )
            elif policy:
                max_norm_error = max_norm(self.q_table, q_star)
                max_norm_error_policy = max_norm(self.q_table, q_star, policy=policy)
            else:
                max_norm_error = max_norm(self.q_table, q_star)
                max_norm_error_policy = 0

            # Almacenar el error de la norma máxima
            self.max_norm_error[episode] = max_norm_error
            self.max_norm_error_policy[episode] = max_norm_error_policy
            # Almacenar cantidad promedio de valores q y del episodio
            self.scores[episode] = total_score
            self.avg_scores[episode] = total_score / max(self.steps[episode], 1)

            # Mostrar información de la ejecución
            message = f"Episodio {episode + 1}/{num_episodes} - Puntaje: {total_score:.2f} - Pasos: {self.steps[episode]} - Max norm error: {max_norm_error:.3f} - Max norm error policy: {max_norm_error_policy:.3f}"
            stqdm.write(message)

    def best_path(self, state):
        """Devuelve el mejor camino desde un estado inicial hasta el estado terminal

        Parameters

        state: int
            estado inicial desde el cual se quiere encontrar el mejor camino hasta el estado terminal
        """
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

    def plot_steps_per_episode(self):
        """
        Grafica la cantidad de pasos que tardó cada episodio en llegar a un estado terminal.
        """
        import matplotlib.pyplot as plt

        plt.figure(dpi=100)
        plt.plot(range(self.num_episodes), self.steps)
        plt.title(self.strategy)
        plt.xlabel("Episodes")
        plt.ylabel("Steps")
        plt.grid()
        plt.show()
