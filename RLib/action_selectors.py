import numpy as np
from typing import Any, Optional
from math import log, sqrt


class ActionSelector(object):
    def __init__(self, **kwargs):
        """Clase base para los selectores de acción. Los selectores de acción son responsables de seleccionar la acción que el agente debe realizar en un estado dado."""
        self.params = kwargs
        self.strategy = kwargs.get("strategy", "default")

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
    def __init__(self, epsilon=0.1):
        """ Selector de acción ε-greedy. La probabilidad de exploración es ε y la probabilidad de explotación es 1-ε.

        Parameters
        ------------
        epsilon: float
            Probabilidad de exploración, valor entre 0 y 1.

        Raises
        ------
        ValueError
            Si el valor de epsilon no está en el rango [0 - 1]

        Examples
        --------
        >>> selector = EpsilonGreedyActionSelector(epsilon=0.1)
        """
        super().__init__(epsilon=0.1, strategy="e-greedy")
        if epsilon < 0 or epsilon > 1:
            raise ValueError("Epsilon debe ser un valor entre 0 y 1")
        self.epsilon = epsilon  # equivalent to kwargs.pop('epsilon', 0.1)

    def select_action(self, agent, state):
        greedy_action = agent.argmax_q_table(state)
        explore_action = agent.random_action(state)
        num = np.random.random()  # Se genera un número aleatorio entre 0 y 1
        # Se explora con probabilidad epsilon y se explota con probabilidad 1-epsilon
        return explore_action if num <= self.epsilon else greedy_action

    def get_label(self):
        return f"ε = {self.epsilon}"


class EpsilonGreedyDecayActionSelector(ActionSelector):
    def __init__(self, constant=0.99):
        r"""Clase para un selector de acción ε-greedy con exploración decreciente. 
        La probabilidad de exploración disminuye a medida que se visitan los estados. 
        La probabilidad de exploración para un estado s está dada por:
        ϵₜ(s) = c / Nₜ(s), donde Nₜ(s) es el número de visitas al estado s en el tiempo t. 
        La constante `c=constant` es un valor entre 0 y 1.

        Parameters
        ------------
        constant: float
            Valor entre 0 y 1 de la constante de la tasa de decrecimiento de `ϵₜ`  

        Examples
        --------
        >>> selector = EpsilonGreedyDecayActionSelector(constant=1)   
        """
        super().__init__(constant=constant, strategy="e-decay")
        self.constant = constant

    def select_action(self, agent, state):
        t = agent.visits_states[state]
        epsilon = self.constant / (t + 1)
        greedy_action = agent.argmax_q_table(state)
        explore_action = agent.random_action(state)
        num = np.random.random()  # Se genera un número aleatorio entre 0 y 1
        # Se explora con probabilidad epsilon y se explota con probabilidad 1-epsilon
        return explore_action if num <= epsilon else greedy_action

    def get_probabilities(self, agent, state):
        """Return the probabilities of selecting each action in the state. Used for debugging purposes"""
        t = agent.visits_states[state]
        epsilon = self.constant / (t + self.constant)
        actions = agent.action_set(state)
        greedy_action = agent.argmax_q_table(state)
        probabilities = dict(zip(actions, [epsilon/len(actions) if action !=
                             greedy_action else 1 - epsilon + epsilon/len(actions) for action in actions]))
        return probabilities

    def get_label(self):
        return f"c = {self.constant}"


class BubeckDecayEpsilonGreedyActionSelector(ActionSelector):
    """
    Extraído de: Bubeck 2012, "Regret Analysis of Stochastic and Nonstochastic Multi-armed Bandit Problems" 2.4.5
    """

    def __init__(self, c: float = 1, d: float = 0.9):
        """
        Similar a epsilon-greedy, pero la probabilidad de exploración disminuye con el tiempo. La probabilidad de exploración está dada por:
            ϵₜ = min{ 1, c A(s) / d² N(s) }
        donde N(s) es la cantidad de visitas al estado s, c y d son constantes de exploración y A(s) es el número de acciones posibles desde el estado.

        Parameters
        ------------
            c : float  
                parametro de exploración, entre 0 y 1

            d: parametro de exploración, entre 0 y 1
        """
        super().__init__(c=c, d=d, strategy="e-decay-Bubeck")
        self.c = c
        self.d = d

    def select_action(self, agent, state):
        n_s = agent.visits_states[state]
        A = len(agent.action_set(state))
        epsilon = min(
            1, self.c * A / (self.d**2 * n_s)
        )  # Epsilon decrece en función de la cantidad de visitas al estado
        greedy_action = agent.argmax_q_table(state)
        explore_action = agent.random_action(state)
        num = np.random.random()  # Se genera un número aleatorio entre 0 y 1
        # Se explora con probabilidad ϵ y se explota con probabilidad 1-ϵ
        return explore_action if num <= epsilon else greedy_action

    def get_label(self):
        return f"ε = min(1, {self.c} * K / ({self.d}^2 * t))"


class UCB1ActionSelector(ActionSelector):
    def __init__(self, c: float = 2):
        """ Selector de acción UCB1. 

        La probabilidad de exploración es proporcional a la raíz cuadrada del logaritmo del tiempo y el número de visitas a un estado.
        Se selecciona la acción que maximiza el intervalo de confianza superior de la recompensa esperada. El intervalo de confianza superior está dado por:

        UCB1(s, a) = Q(s, a) + c * sqrt(log(t) / N(s, a))

        Parameters
        ------------
        c: float
            Parámetro de exploración, valor mayor a 0.
        """
        super().__init__(c=2, strategy="UCB1")
        c < 0 and ValueError("El valor de c debe ser mayor a 0")
        self.c = c
        self.strategy = f"UCB1"

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

    def __init__(self, c: float = 2):
        """ Inicializar el selector de acción AsOpt-UCB. La probabilidad de exploración es proporcional a la raíz cuadrada del logaritmo del tiempo y el número de visitas a un estado.
        Se selecciona la acción que maximiza el intervalo de confianza superior de la recompensa esperada. Este selector es asintóticamente óptimo en un entorno de Bandits, es decir, en el Multi Armed Bandit converge a la política óptima a medida que el número de visitas a un estado tiende a infinito.
        
        El intervalo de confianza superior está dado por:
        
        UCB(s, a) = Q(s, a) + c * sqrt(f(t) / N(s, a))
        donde f(t) = 1 + t * log(t)²
        donde t:= N(s) es el número de visitas al estado s y N(s, a) es el número de visitas a la acción a en el estado s.
        

        Parameters
        ------------
        c: float
            Exploration parameter for the UCB formula. UCB(s, a) = Q(s, a) + c * sqrt(log(t) / N(s, a))
        
        Examples
        --------
        >>> selector = AsOptUCBActionSelector(c=2)
        """
        super().__init__(c=2, strategy="AsOpt-UCB")
        self.c = c

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
            t = visits_state + 1
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

    def __init__(self, eta: str = "log(n_s) / q_range"):
        """Initialize the Exp3 action selector.

        Parameters
        ----------
        eta : str, optional
            Parámetro de exploración, puede ser constante o una expresión que dependa del tiempo.

        Examples
        --------
        >>> allowed_variables = ['t', 'T', 'A', 'n_s', 'q_range']
        >>> eta_values = ['sqrt(t)', 'log(t+1)', '0.1', 'log( n_s ) * n_s**(1/2) / q_range', 'log( n_s ) / q_range', 'sqrt( n_s ) / q_range']
        >>> eta = "log(n_s) / q_range"
        >>> selector = Exp3ActionSelector(eta=0.1)
        >>> action = selector.select_action(agent, state)
        >>> selector = Exp3ActionSelector(eta="sqrt(t)")
        >>> action = selector.select_action(agent, state)
        """
        super().__init__(eta=eta, strategy="exp3")
        self.eta = str(eta)

    def calculate_probabilities(self, q_values, eta):
        # Los valores de q se normalizan para evitar problemas de overflow (restando el máximo valor de q)
        max_q = np.max(q_values)
        exp_values = np.exp((q_values - max_q) * eta)
        # Calcular probabilidades de selección de acciones
        probabilities = exp_values / np.sum(exp_values)
        self.probabilities = probabilities
        return probabilities

    def get_probabilities(self, agent, state):
        """Return the probabilities of selecting each action in the state. Used for debugging purposes"""
        return dict(zip(agent.action_set(state), self.probabilities))

    def select_action(self, agent, state) -> Any:
        """Select an action using the Exp3 algorithm.

        Parameters
        ----------
        agent : QAgentSSP
            Agente que selecciona la acción
        state : int
            Estado actual

        Returns
        -------
        Any
            Acción seleccionada

        Raises
        ------
        ValueError
            Si no se puede seleccionar una acción en el estado actual
        """

        # Visitas al estado actual
        t = agent.actual_episode
        # Número total de episodios (Horizonte de tiempo)
        T = agent.num_episodes
        # acciones disponibles en el estado actual
        actions = agent.action_set(state)
        q_values = np.fromiter(
            (agent.q_table[state][action] for action in actions), dtype=float)
        q_value_range = np.ptp(q_values)  # peak-to-peak value (max - min)
        # Evaluar eta en función del tiempo y el número de episodios
        params = {'t': t, 'T': T,
                  'sqrt': sqrt, 'log': log,
                  'A': len(actions),
                  'n_s': agent.visits_states[state] + 1,
                  'q_range': max(q_value_range, 0.001),
                  }
        eta = eval(self.eta, params)
        # Calcular probabilidades de selección cada acción
        probabilities = self.calculate_probabilities(q_values, eta)
        try:
            # Se utiliza el índice para evitar problemas de tipo de datos en el muestreo
            action_idx = np.random.choice(len(actions), p=probabilities)
            return actions[action_idx]
        except ValueError:
            raise ValueError(
                f"Error al seleccionar acción en estado {state}. Probabilidades: {probabilities}. Acciones: {actions}"
            )

    def get_label(self):
        return f"η = {self.eta}"


if __name__ == "__main__":

    print("Ejemplos de uso de los selectores de acción:")

    selector1 = Exp3ActionSelector(beta="t / T")
    selector2 = UCB1ActionSelector(c=2)
    selector3 = EpsilonGreedyActionSelector(epsilon=0.1)

    print(selector1, selector2, selector3, sep="\n")
