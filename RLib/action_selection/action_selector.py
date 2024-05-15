import numpy as np
from math import log, sqrt


def auto_super_init(func):
    def wrapper(self, *args, **kwargs):
        args_locals = func(self, *args, **kwargs) or {}
        # print(f"Variables locales antes de eliminar 'self' y '__class__': {args_locals}")
        args_locals.pop("self", None)
        args_locals.pop("__class__", None)
        # print(f"Variables locales después de la limpieza: {args_locals}")
        super(self.__class__, self).__init__(**args_locals)

    return wrapper


class ActionSelector(object):
    """Abstract class for action selection strategies"""

    def __init__(self, **kwargs):
        self.params = kwargs

    def select_action(self, agent, state):
        raise NotImplementedError()

    def __str__(self):
        return self.__class__.__name__ + str(self.params)

    def get_label(self):
        return f"{self.__class__.__name__.replace('ActionSelector', '')}"


class EpsilonGreedyActionSelector(ActionSelector):
    @auto_super_init
    def __init__(self, epsilon=0.1):
        """epsilon: probabilidad de seleccionar una acción aleatoria en lugar de la mejor acción según Q(s,a)"""
        self.epsilon = epsilon  # equivalent to kwargs.pop('epsilon', 0.1)
        self.strategy = "e-greedy"
        return locals()  # Dictionary with the local variables

    def select_action(self, agent, state):
        greedy_action = agent.argmax_q_table(state)
        explore_action = agent.random_action(state)
        num = np.random.random()  # Se genera un número aleatorio entre 0 y 1
        # Se explora con probabilidad epsilon y se explota con probabilidad 1-epsilon
        return explore_action if num <= self.epsilon else greedy_action

    def get_label(self):
        return f"ε = {self.epsilon}"


class DynamicEpsilonGreedyActionSelector(ActionSelector):
    """
    Extracted from: Bubeck 2012, "Regret Analysis of Stochastic and Nonstochastic Multi-armed Bandit Problems" 2.4.5
    """

    @auto_super_init
    def __init__(self, c, d):
        """
        Similar to epsilon-greedy, but the probability of exploration decreases over time. The probability of exploration is given by:
            epsilon(t) = min (1, c * K / (d ** 2 * t))
        where t is the time step, c and d are constants.

        Parameters:
            c: exploration parameter, greater than 0
            d: exploration parameter, between 0 and 1
        """
        self.c = c
        self.d = d
        self.strategy = "e-decay"
        return locals()

    def select_action(self, agent, state):
        t = agent.times_states[state]
        epsilon = min(
            1, self.c * agent.num_actions / (self.d**2 * t)
        )  # Epsilon decreases over time
        greedy_action = agent.argmax_q_table(state)
        explore_action = agent.random_action(state)
        num = np.random.random()  # Se genera un número aleatorio entre 0 y 1
        # Se explora con probabilidad epsilon y se explota con probabilidad 1-epsilon
        return explore_action if num <= epsilon else greedy_action

    def get_label(self):
        return f"ε = min(1, {self.c} * K / ({self.d}^2 * t))"


class UCB1ActionSelector(ActionSelector):
    @auto_super_init
    def __init__(self, c=2):
        """Parámetros
        c: parámetro de exploración
        """
        self.c = c
        self.strategy = "UCB1"
        return locals()

    def select_action(self, agent, state):
        c = self.c
        # Lista de acciones disponibles en el estado actual
        actions = list(agent.q_table[state].keys())
        # Contador de visitas para cada accion en el estado actual
        times_actions = np.array(list(agent.times_actions[state].values()))
        # Contar la cantidad de acciones que no han sido escogidas en el estado actual
        not_chosen_actions_idx = np.where(times_actions == 0)[0]
        if not_chosen_actions_idx.shape[0] > 0:
            # Escoger una acción no visitada anteriormente de forma aleatoria
            action_idx = np.random.choice(
                not_chosen_actions_idx  # Ex: [0, 1, 2, 3]
            )  # Se escoge el índice de la acción
            action = list(agent.times_actions[state].keys())[
                action_idx
            ]  # Se obtiene la acción a partir del índice
        else:
            # Obtener los valores de q para cada acción a partir del estado s
            q_state = np.array(list(agent.q_table[state].values()))
            # Calcular el valor de los estimadores de q utilizando la estrategia UCB
            ucb = q_state + np.sqrt(c * np.log(agent.num_episodes) / times_actions)
            # Seleccionar acción
            action_idxs = np.where(ucb == np.max(ucb))[0]
            action_idx = np.random.choice(action_idxs)
            action = list(agent.q_table[state].keys())[action_idx]
        # Devolver acción
        return action

    def get_label(self):
        return f"c = {self.c}"


class Exp3ActionSelector(ActionSelector):
    """Exp3 action selector"""

    @auto_super_init
    def __init__(self, beta):
        """
        Exp3 action selector tiene un parámetro beta que puede ser constante o dinámico. En el caso de ser constante se debe ingresar un valor de beta, en el caso de ser dinámico se debe ingresar una fórmula para calcular beta en función del tiempo y el número de episodios.

        Ejemplos: (t: instante de tiempo, T: número de episodios)
            - beta = 0.1
            - beta = 't / T'
            - beta = 'log(t)'
            - beta = 'sqrt(t)'
            - beta = 'sqrt(t) / T'
            - beta = 'log(t) / T'

        Parámetros:
            beta: parámetro de exploración
        """

        self.beta_formula = beta
        self.beta_type = "constant" if type(beta) == float else "dynamic"
        self.strategy = "exp3"
        return locals()

    def select_action(self, agent, state, normalize=True):
        t = agent.times_states[state]
        T = agent.num_episodes

        beta = (
            eval(self.beta_formula)
            if self.beta_type == "dynamic"
            else self.beta_formula
        )

        # Obtener los valores de q para cada acción a partir del estado s
        q_state = np.array(list(agent.q_table[state].values()))
        q_actions = list(agent.q_table[state].keys())
        actions_idx = np.arange(len(q_actions))
        exp_values = (
            np.exp((q_state - np.max(q_state)) * beta)
            if normalize
            else np.exp(q_state * beta)
        )
        # Generar distribucion de probabilidad exponencial
        probabilities = exp_values / np.sum(exp_values)
        # Muestrear acción de acuerdo a la distribución generada
        try:
            selected_action_idx = np.random.choice(actions_idx, p=probabilities)
            action = q_actions[selected_action_idx]
        except ValueError:
            raise ValueError(
                f"Error al seleccionar acción en estado {state}. Probabilidades: {probabilities}. Acciones: {q_actions}"
            )
        return action

    def get_label(self):
        return f"η = {self.beta_formula}"


if __name__ == "__main__":

    print("Ejemplos de uso de los selectores de acción:")

    selector1 = Exp3ActionSelector(beta="t / T")
    selector2 = UCB1ActionSelector(c=2)
    selector3 = EpsilonGreedyActionSelector(epsilon=0.1)

    print(selector1, selector2, selector3, sep="\n")
