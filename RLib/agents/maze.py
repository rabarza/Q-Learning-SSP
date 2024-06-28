import numpy as np
import matplotlib.pyplot as plt
import random
import copy
from tqdm import tqdm
from RLib.utils.dijkstra import max_norm, exploitation


class Q_maze:
    """
    Resuelve laberintos empleando el algoritmo Q-Learning. Se consideran dos casos. El primer caso considera un laberinto compuestos por fosas, es decir, cuando el agente cae en una fosa se muere y finaliza el episodio, notar que tambien finaliza cuando llega a la meta. El segundo caso es un laberinto con paredes de fuego, el fuego no mata al agente, pero le hace daño al quemarlo, por lo que recibe un castigo cada vez que toca el fuego, el episodio termina exclusivamente cuando el agente llega a la meta, sin importar cuánto se haya quemado.


    #### Parámetros:

    `rewards` --  matriz de recompensas que compone la estructura del laberinto.

    `episodes` -- cantidad de iteraciones que se empleará en el algoritmo Q-Learning.

    `gamma` -- tasa de descuento.

    `strategy` -- estrategia de exploracion-explotacion, puede ser e-greedy o UCB1.

    `epsilon` -- valor de epsilon que se emplea cuando se utiliza una estrategia e-greedy (por defecto epsilon = 0.1).

    `game` -- tipo de laberinto compuesto por fosas, o por paredes de fuego (pit-walls o fire-walls, respectivamente).

    """

    def __init__(
        self,
        environment,
        strategy="e-greedy",
        epsilon=0.1,
        alpha=0.8,
        gamma=0.9,
        game="fire-walls",
        temperature=1,
        c=2,
        d=0.8,
        expression="t^2",
    ):

        self.env = environment
        self.num_states = np.prod(environment.maze.shape)
        self.num_actions = 4

        self.strategy = strategy
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.game = game

        # Se cuenta la cantidad de veces que se tomo una accion en cada estado
        self.times_actions = np.zeros((self.num_states, self.num_actions))

        # Se cuenta la cantidad de veces que se visita un estado
        self.times_states = np.zeros(self.num_states)

        # Se mide los pesos y probabilidades de tomar cada acción
        self.action_weights = np.ones(self.num_actions)
        self.action_prob = np.zeros(self.num_actions)

        # e-decay
        self.c = c
        self.d = d

        # exp3
        self.expression = expression

        # boltzmann
        self.tau = temperature

    def select_action(self, state):
        """Devuelve una acción de acuerdo a la estrategia de selección de acción.

        #### Parámetros:

        `state` -- estado desde el cual se realiza la accion.

        `strategy` --  método que se usará para explorar-explotar.

        `epsilon` -- solo se utiliza en el método epsilon-greedy (por defecto epsilon = 0).

        """

        def epsilon_greedy(self, state, epsilon):
            greedy_action = np.argmax(self.q_table[state])
            explore_action = np.random.randint(self.num_actions)

            num = np.random.random()
            if num <= epsilon:
                # Explorar
                return explore_action
            else:
                # Explotar
                return greedy_action

        def epsilon_truncated(self, state, epsilon):
            greedy_action = np.argmax(self.q_table[state])
            explore_action = np.random.randint(self.num_actions)

            if self.steps[-1] >= 125:
                epsilon = 0

            num = np.random.random()
            if num <= epsilon:
                # Explorar
                return explore_action
            else:
                # Explotar
                return greedy_action

        def decreasing_epsilon_greedy(self, state):
            K = self.num_actions
            c = self.c
            d = self.d

            # Incrementar cantidad de visitas al estado
            self.times_states[state] += 1
            t = self.times_states[state]

            greedy_action = np.argmax(self.q_table[state])
            explore_action = np.random.randint(self.num_actions)

            # Actualizar valor de la tasa epsilon_n decreciente
            epsilon_n = min(1, (c * K) / ((d**2) * t))

            # Seleccionar acción
            num = np.random.random()
            if num <= epsilon_n:
                # Explorar
                return explore_action
            else:
                # Explotar
                return greedy_action

        def softmax_exploration(self, state, normalize=True):
            # Incrementar cantidad de visitas al estado
            self.times_states[state] += 1

            # Actualizar el valor de temperatura
            tau = self.tau

            q_state = self.q_table[state]

            if normalize:
                exp_values = np.exp((q_state - np.max(q_state)) * tau)
            else:
                exp_values = np.exp(q_state * tau)

            # Generar distribucion de probabilidad exponencial
            probabilities = exp_values / np.sum(exp_values)

            # Muestrear acción de acuerdo a la distribución generada
            action = np.random.choice(np.arange(self.num_actions), p=probabilities)
            return action

        def upper_confidence_bound(self, state, c=2):
            # Contador de visitas para cada accion en el estado actual
            times_actions = self.times_actions[state]

            # Contar la cantidad de acciones que no han sido escogidas en el estado actual
            number_not_chosen_actions = np.count_nonzero(times_actions == 0)

            if number_not_chosen_actions > 0:
                # Escoger una acción no visitada anteriormente
                action = np.argwhere(times_actions == 0).flatten()[0]
                # Incrementar el contador de visitas para la acción en el estado actual
                self.times_actions[state, action] += 1

                # Devolver acción
                return action

            else:
                # Obtener los valores de q para cada acción a partir del estado s
                q_state = self.q_table[state]

                # Calcular el valor de los estimadores de q utilizando la estrategia UCB
                ucb = q_state + np.sqrt(c * np.log(self.num_episodes) / times_actions)

                # Seleccionar acción
                action = np.argmax(ucb)

                # Incrementar el contador de visitas para la acción en el estado actual
                self.times_actions[state, action] += 1

                # Devolver acción
                return action

        def exp3_action_selection(self, state, normalize=True):

            # Incrementar cantidad de visitas al estado
            self.times_states[state] += 1
            t = self.times_states[state]

            # Actualizar el valor de temperatura decreciente
            match self.expression:
                case "t":
                    eta = t
                case "1/t":
                    eta = 1 / t
                case "t/T":
                    eta = t / self.num_episodes
                case "t^2":
                    eta = t**2
                case "t^2/T":
                    eta = t**2 / self.num_episodes
                case "t^3":
                    eta = t**3
                case "\log t":
                    eta = np.log(t)
                case "\sqrt{t}":
                    eta = np.sqrt(t)
                case _:
                    raise ValueError("Expresión inválida para eta.")

            # Obtener los valores de q para cada acción a partir del estado s
            q_state = self.q_table[state]

            # Valores exponenciales de Q(s,a)
            if normalize:
                exp_values = np.exp((q_state - np.max(q_state)) * eta)
            else:
                exp_values = np.exp(q_state * eta)

            # Generar distribucion de probabilidad exponencial
            probabilities = exp_values / np.sum(exp_values)

            # Muestrear acción de acuerdo a la distribución generada
            action = np.random.choice(
                np.arange(self.num_actions), p=list(probabilities)
            )
            return action

        match self.strategy:
            case "e-greedy":
                return epsilon_greedy(self, state, self.epsilon)

            case "e-truncated":
                return epsilon_truncated(self, state, self.epsilon)

            case "e-decay":
                return decreasing_epsilon_greedy(self, state)

            case "UCB1":
                return upper_confidence_bound(self, state)

            case "softmax":
                return softmax_exploration(self, state)

            case "exp3":
                return exp3_action_selection(self, state)

            case _:
                raise ValueError(
                    "El método seleccionado debe ser valido (como e-greedy, e-decay, UCB1, softmax)"
                )

    def train(self, num_episodes):
        """
        Resuelve el problema del laberinto usando el algoritmo Q-Learning
        """
        self.num_episodes = num_episodes
        self.steps = np.zeros(num_episodes)
        self.scores = np.zeros(num_episodes)
        self.mean_q_values = np.zeros(num_episodes)
        alpha = self.alpha
        gamma = self.gamma

        # inicializar tabla de valores q
        self.q_table = np.zeros((self.num_states, self.num_actions))

        for episode in range(self.num_episodes):
            total_score = 0
            acum_q_values = 0
            state = self.env.start_state()

            while not self.env.terminal_state(state, self.game):

                # Se realiza la transición (action, next_state, reward)
                action = self.select_action(state)
                next_state, reward = self.env.take_action(state, action)

                # Actualizar valores Q_table
                q_old = self.q_table[state, action]
                # alpha = 1 / (self.times_states[state])
                q_new = q_old * (1 - alpha) + alpha * (
                    reward + gamma * np.max(self.q_table[next_state])
                )

                self.q_table[state, action] = q_new

                # Ir al estado siguiente
                state = next_state

                # Aumentar la cantidad de pasos del episodio
                self.steps[episode] += 1
                # Acumular el puntaje obtenido en el episodio
                total_score += reward
                acum_q_values += q_new

            # Agregar cantidad promedio de valores q y del episodio
            self.mean_q_values[episode] = acum_q_values / max(self.steps[episode], 1)
            self.scores[episode] = total_score / max(self.steps[episode], 1)

    def best_path(self, state):
        """Devuelve una lista que contiene todas las ubicaciones que se deben recorrer para llegar desde un estado a la solución del laberinto realizando la menor cantidad de pasos.

        Parámetros:

        `state` -- estado (debe ser una lista que contenga las cordenadas x, y. ej: [x, y])
        """
        state = self.env.index(state)
        path = []
        if self.env.terminal_state(state, self.game):
            return path

        else:
            path.append(self.env.position(state))

        while not self.env.terminal_state(state, self.game):
            action = np.argmax(self.q_table[state])
            state = self.env.take_action(state, action)[0]
            path.append(self.env.position(state))
        return path

    def plot_steps_per_episode(self):
        """
        Grafica la cantidad de pasos que tardó cada episodio en llegar a un estado terminal.
        """
        import matplotlib.pyplot as plt

        plt.figure(dpi=100)
        plt.plot(range(self.num_episodes), self.steps)
        plt.title(self.game + "-" + self.strategy)
        plt.xlabel("Episodes")
        plt.ylabel("Steps")

        plt.grid()
        plt.show()
