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
    
    def action_set(self, agent, state):
        return agent.action_set(state)

    def action_set(self, agent, state):
        return agent.action_set(state)

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
        t = agent.visits_states[state]
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
        self.strategy = f"UCB1"
        return locals()

    def select_action(self, agent, state):
        c = self.c
        # Lista de acciones disponibles en el estado actual
        actions = agent.action_set(state)
        # Contador de visitas para cada accion en el estado actual
        visits_state = agent.visits_states[state]
        visits_state_action = np.fromiter(
            (agent.visits_actions[state][action] for action in actions), dtype=float)
        # Contar la cantidad de acciones que no han sido escogidas en el estado actual
        not_chosen_actions_idx = np.where(visits_state_action == 0)[0]
        # Si alguna(s) de las acciones aún no ha sido escogida...
        if not_chosen_actions_idx.shape[0] > 0:
            # Se selecciona aleatoriamente esa acción dentro de las no visitadas
            # unselected action index
            action_idx = np.random.choice(not_chosen_actions_idx)
        else:
            # Obtener los valores de q para cada acción a partir del estado s
            q_state = np.fromiter(
                (agent.q_table[state][action] for action in actions), dtype=float)
            # Calcular el valor de los estimadores de q utilizando la estrategia UCB
            t = max(visits_state, 1)
            ucb = q_state + c * np.sqrt(np.log(t) / visits_state_action)
            max_ucb = np.max(ucb)
            # indexes where the max is attained
            max_idxs = np.argwhere(ucb == max_ucb)[0]
            # argmax action index
            action_idx = np.random.choice(max_idxs)
        action = actions[action_idx]
        # Devolver acción
        return action

    def get_label(self):
        return f"c = {self.c}"


class AsOptUCBActionSelector(ActionSelector):
    """Asymptotically Optimal UCB action selector"""
    @auto_super_init
    def __init__(self, c=2):
        """Parámetros
        c: parámetro de exploración
        """
        self.c = c
        self.strategy = f"AsOpt-UCB"
        return locals()

    def select_action(self, agent, state):
        c = self.c
        # Lista de acciones disponibles en el estado actual
        actions = agent.action_set(state)
        # Contador de visitas para cada accion en el estado actual
        visits_state = agent.visits_states[state]
        visits_state_action = np.fromiter(
            (agent.visits_actions[state][action] for action in actions), dtype=float)
        # Contar la cantidad de acciones que no han sido escogidas en el estado actual
        not_chosen_actions_idx = np.where(visits_state_action == 0)[0]

        if not_chosen_actions_idx.shape[0] > 0:
            # Escoger una acción no visitada anteriormente de forma aleatoria
            action_idx = np.random.choice(
                not_chosen_actions_idx  # Ex: [0, 1, 2, 3]
            )  # Se escoge el índice de la acción
        else:
            # Obtener los valores de q para cada acción a partir del estado s
            q_state = np.fromiter(
                (agent.q_table[state][action] for action in actions), dtype=float)
            # Calcular el valor de los estimadores de q utilizando la estrategia UCB
            t = max(visits_state, 1)
            f_t = 1 + t * np.log(t)**2
            ucb = q_state + np.sqrt(c * np.log(f_t) / visits_state_action)
            # Seleccionar acción
            max_ucb = np.max(ucb)
            # indexes where the max is attained
            max_idxs = np.argwhere(ucb == max_ucb)[0]
            # argmax action index
            action_idx = np.random.choice(max_idxs)
        action = actions[action_idx]
        # Devolver acción
        return action

    def get_label(self):
        return f"c = {self.c}"


class Exp3ActionSelector(ActionSelector):
    """Exp3 action selector"""

    @auto_super_init
    def __init__(self, eta):
        """
        Exp3 action selector tiene un parámetro eta que puede ser constante o dinámico. En el caso de ser constante se debe ingresar un valor de eta, en el caso de ser dinámico se debe ingresar una fórmula para calcular eta en función del tiempo y el número de episodios.

        Ejemplos: (t: instante de tiempo, T: número de episodios)
            - eta = 0.1 (o cualquier valor de eta constante de tipo float)
            - eta = 't / T'
            - eta = 'log(t)'
            - eta = 'sqrt(t)'
            - eta = 'sqrt(t) / T'
            - eta = 'log(t) / T'

        Parámetros:
            eta: parámetro de exploración
        """

        self.eta = str(eta)
        self.strategy = "exp3"
        return locals()

    def calculate_probabilities(self, agent, state, eta):
        # acciones disponibles en el estado actual
        actions = agent.action_set(state)
        # Obtener los valores de q para cada acción a partir del estado s
        q_state = np.fromiter(
            (agent.q_table[state][action] for action in actions), dtype=float)
        # Los valores de q se normalizan para evitar problemas de overflow (restando el máximo valor de q)
        max_q = np.max(q_state)
        exp_values = np.exp((q_state - max_q) * eta)
        # Calcular probabilidades de selección de acciones
        probabilities = exp_values / np.sum(exp_values)
        return probabilities

    def select_action(self, agent, state):
        # Visitas al estado actual
        t = agent.visits_states[state]
        # Número total de episodios (Horizonte de tiempo)
        T = agent.num_episodes
        # Calcular probabilidades de selección de acciones
        actions = agent.action_set(state)
        actions_idx = np.arange(len(actions))
        # Evaluar eta en función del tiempo y el número de episodios
        eta = eval(self.eta, {'t': t, 'T': T, 'sqrt': sqrt,
                   'log': log, 'A': len(actions)})
        probabilities = self.calculate_probabilities(agent, state, eta)
        try:
            # Muestrear acción de acuerdo a la distribución generada
            # Se utiliza el índice para evitar problemas de tipo de datos en el muestreo
            action_idx = np.random.choice(
                actions_idx, p=probabilities)
            action = actions[action_idx]
        except ValueError:
            raise ValueError(
                f"Error al seleccionar acción en estado {state}. Probabilidades: {probabilities}. Acciones: {actions}"
            )
        return action

    def get_label(self):
        return f"η = {self.eta}"


if __name__ == "__main__":

    print("Ejemplos de uso de los selectores de acción:")

    selector1 = Exp3ActionSelector(beta="t / T")
    selector2 = UCB1ActionSelector(c=2)
    selector3 = EpsilonGreedyActionSelector(epsilon=0.1)

    print(selector1, selector2, selector3, sep="\n")
