import random
import numpy as np
import networkx as nx
import osmnx as ox
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from RLib.distributions.distributions import LogNormalDistribution, expected_time, random_time


class Grafo:
    '''Clase que representa un grafo con un número de nodos dado y una densidad dada
    '''
    def __init__(self, num_nodos=None, adj_matrix=None):
        '''Constructor de la clase Grafo
        Args:
            num_nodos (int): Número de nodos del grafo
        '''
        self.__num_nodos = num_nodos if num_nodos is not None else adj_matrix.shape[0]
        self.__density = 0.5
        self.__adjacency_matrix = adj_matrix if adj_matrix is not None else self.generar_matriz_adyacencia(self.__density)
        self.__params_matrix = self.gen_params_matrix()

    @property
    def num_nodos(self):
        return self.__num_nodos
    @property
    def adjacency_matrix(self):
        return self.__adjacency_matrix
    @property
    def density(self):
        return self.__density
    @property
    def params_matrix(self):
        return self.__params_matrix
    @density.setter
    def density(self, density):
        self.__density = density
        
    def __str__(self):
        return f'Grafo con {self.num_nodos} nodos y densidad {self.density}'
    
    def generar_matriz_adyacencia(self, density=0.8):
        '''Generar una matriz de adyacencia aleatoria para un grafo con un número de nodos dado
        Parámetros:
            num_nodos (int): Número de nodos del grafo
        Retorno:
            adjacency_matrix (np.array): Matriz de adyacencia del grafo
        '''
        adjacency_matrix = np.zeros((self.num_nodos, self.num_nodos))
        
        # for i in range(self.num_nodos - 1):
        #     if random.random() < density:

        #         num_conexiones = np.random.randint(1, self.num_nodos - i)
        #         conexiones = np.random.choice(self.num_nodos - i - 1, size=num_conexiones, replace=False) + i + 1

        #         adjacency_matrix[i, conexiones] = 1
        #         adjacency_matrix[conexiones, i] = 1  # Asignar conexión simétrica
        for i in range(self.num_nodos):
            for j in range(i+1, self.num_nodos-1):
                
                if random.random() < density:
                    adjacency_matrix[i][j] = adjacency_matrix[j][i] = 1

        k = random.randint(1, self.num_nodos-2)  # Número de nodos intermedios entre el nodo inicial y el nodo final (por eso se resta 2)
        nodos_intermedios = random.sample(range(1, self.num_nodos-1), k)  # Selección aleatoria de k nodos intermedios

        # Conexiones en secuencia entre los nodos intermedios
        for i in range(k-1):
            adjacency_matrix[nodos_intermedios[i]][nodos_intermedios[i+1]] = adjacency_matrix[nodos_intermedios[i+1]][nodos_intermedios[i]] = 1
        
        # Conexiones entre el nodo inicial y el primer nodo intermedio
        adjacency_matrix[0][nodos_intermedios[0]] = adjacency_matrix[nodos_intermedios[0]][0] = 1

        # Conexiones entre el último nodo intermedio y el nodo final
        adjacency_matrix[nodos_intermedios[-1]][self.num_nodos-1] = adjacency_matrix[self.num_nodos-1][nodos_intermedios[-1]] = 1

        return adjacency_matrix
    
    def gen_params_matrix(self, mean=0, std=5, gamma_shape=2, gamma_scale=3):
        '''
        Genera una matriz de parámetros para las distribuciones de recompensas de los arcos
        
        # Parámetros:
            mean (float): Media de la distribución lognormal
            std (float): Desviación estándar de la distribución lognormal
        Retorno:
            params_matrix (np.array): Matriz de parámetros de las distribuciones de recompensas de los arcos
        '''
        params_matrix = [[(np.random.normal(mean, std), np.random.gamma(gamma_shape, gamma_scale)) for i in range(self.num_nodos)] for j in range(self.num_nodos)]
        return params_matrix
    
    def reset_params_matrix(self):
        '''	
        Reinicia la matriz de parámetros de las distribuciones de recompensas de los arcos del grafo
        '''
        self.__params_matrix = self.gen_params_matrix()

    
    def obtener_costo_arco(self, origen:int, destino:int):
        '''Obtener el costo de un arco entre dos nodos
        # Parámetros:
            origen (int): Nodo de origen
            destino (int): Nodo de destino
            Retorno:
            costo (float): Costo del arco entre los nodos origen y destino (es siempre positivo)
        '''	

        if self.adjacency_matrix[origen][destino] == 1: # Si hay un arco entre los nodos origen y destino
            mu = self.__params_matrix[origen][destino][0]
            sigma = self.__params_matrix[origen][destino][1]
            return np.random.lognormal(mu, sigma)
        else: 
            return 10000  # Penalidad por intentar cruzar un arco que no existe
            # raise ValueError(f'No existe un arco entre los nodos {origen} y {destino}')

    def render(self, labeled=False):
        '''Graficar el grafo con las distribuciones de recompensas
        '''
        G = nx.Graph() # Crear grafo dirigido (DiGraph), grafo no dirigido (Graph) o multigrafo dirigido (MultiDiGraph)

        # Agregar nodos al grafo
        [G.add_node(i) for i in range(self.num_nodos)]

        # Agregar arcos al grafo con las distribuciones de recompensas
        for i in range(self.num_nodos):
            for j in range(self.num_nodos):
                if self.adjacency_matrix[i][j] == 1: # Si hay un arco entre los nodos i y j
                    distribution = f'N({self.__params_matrix[i][j][0]:.2f}, {self.__params_matrix[i][j][1]:.2f})' if labeled else ''
                    G.add_edge(i, j, label=distribution) 

        # Definir posición de los nodos en el gráfico
        pos = nx.circular_layout(G) 

        # Dibujar el grafo
        nx.draw_networkx(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10)

        # Mostrar etiquetas de las distribuciones en los arcos
        labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=8)

        # Mostrar el gráfico
        plt.title('Grafo con Distribuciones de Recompensas')
        plt.axis('off')
        plt.show()
        


class EnvShortestPath:
    '''Entorno de aprendizaje para el problema de encontrar el camino más corto en un grafo	
    '''
    def __init__(self, grafo):
        '''Constructor de la clase	
        # Parámetros:
            grafo (Grafo): Grafo con el que se va a trabajar
        '''
        self.grafo = grafo
        self.num_nodos = grafo.num_nodos
        self.__start_state = 0
        self.__current_state = self.__start_state
        self.__terminal_state = self.num_nodos - 1
        self.__num_states = self.grafo.num_nodos
        self.__num_actions = self.grafo.num_nodos

    def __str__(self):
        return f'EnvShortestPath {self.grafo}'
    @property
    def num_states(self):
        return self.__num_states
    @property
    def num_actions(self):
        return self.__num_actions
    @property
    def start_state(self):
        return self.__start_state
    @property
    def terminal_state(self):
        return self.__terminal_state
    @property
    def current_state(self):
        return self.__current_state
    
    def reset(self):
        '''Reiniciar el entorno de aprendizaje'''
        self.__current_state = self.__start_state
        self.grafo.reset_params_matrix()
        print(f'Entorno reiniciado. Grafo con {self.num_nodos} nodos. Estado inicial: {self.__start_state}. Estado final: {self.__terminal_state}')

    def take_action(self, state, action):
        '''Tomar una acción en el entorno de aprendizaje
        Parámetros:
            state (int): Estado actual
            action (int): Acción a tomar
        Retorno:
            next_state (int): Siguiente estado
            reward (float): Recompensa (o costo) por tomar la acción
            terminated (bool): Indica si el episodio terminó
        '''

        links = np.where(self.grafo.adjacency_matrix[state] == 1)[0] # [0] (np.where devuelve un array con los indices de los elementos que cumplen la condicion) pero en una tupla y con [0] se obtiene el primer elemento de la tupla que es el array con los indices

        if action in links:  # Moverse a la siguiente posición en el grafo
            next_state = action # El siguiente estado es el nodo al que se mueve
            costo = self.grafo.obtener_costo_arco(self.current_state, next_state)
            
            if next_state == self.terminal_state:
                terminated = True
                reward = 100000  # Recompensa alta al llegar al nodo final
            else:
                terminated = False
                reward = -costo  # Penalidad por el costo del arco
        else:  # Acción inválida (no hay arco entre los nodos) y se queda en el mismo nodo
            next_state = state
            terminated = False
            reward = -10000  # Penalidad por intentar cruzar un arco que no existe
            self.nodo_actual = next_state
            info = { 'recompensa': reward, 'terminado': terminated}
            return next_state, reward, terminated, info
        self.nodo_actual = next_state
        info = { 'recompensa': reward, 'terminado': terminated}
        return next_state, reward, terminated, info
    
##############################################################################################################
# Stochastic Shortest Path
##############################################################################################################
def get_edge_attribute(G, orig, dest, weight="length"):
    edge_data = G.get_edge_data(orig, dest)
    if weight in edge_data:
        edge_length = edge_data[weight]
    else:
        edge_features = next(iter(edge_data.values()))
        assert weight in edge_features, f'No hay atributo {weight} en los arcos entre los nodos {orig} y {dest}.'
        edge_length = edge_features[weight]    
    return edge_length

def get_edge_length(G, orig, dest):
    assert G.has_edge(orig, dest), f'No hay arco entre los nodos {orig} y {dest}.'
    return get_edge_attribute(G, orig, dest, "length")

def get_edge_speed(G, orig, dest, avg_speed=25):
    try:
        # edge_speed = get_edge_attribute(G, orig, dest, "speed_kph")
         edge_speed = avg_speed
    except Exception:
        edge_speed = avg_speed
    if (edge_speed - avg_speed > 10):
        edge_speed = avg_speed
    return edge_speed

def get_edge_cost(G, origen:int, destino:int, distribution_name:str='expectation-lognormal', avg_speed=25):
    '''### Obtener el costo de un arco entre dos nodos
    Parámetros:
        G (Grafo): Grafo con el que se va a trabajar
        origen (int): Nodo origen
        destino (int): Nodo destino
        distribution_name (str): Distribución de probabilidad que se va a utilizar para obtener el costo estocástico. Si no se especifica, el costo es determinístico y se utiliza el valor de la esperanza de la distribución lognormal.
    Retorno:
        stochastic_cost (float): Costo del arco entre los nodos origen y destino
    '''
    edge_length = get_edge_length(G, origen, destino)
    edge_speed = get_edge_speed(G, origen, destino, avg_speed)
        
    # stochastic cost sample from lognormal distribution_name
    if distribution_name == 'expectation-lognormal':
        # get the expected speed and calculate the expected time given the edge length
        time = expected_time(edge_length, edge_speed)
        
    elif distribution_name == 'lognormal':
        # generate a sample of the speed and calculate the time given the edge length
        time = random_time(edge_length, edge_speed)
    else:
        raise ValueError(f'La distribución {distribution_name} no está implementada.')
    stochastic_cost = time
    return stochastic_cost

def get_cumulative_edges_cost(grafo, policy, node, dest_node):
    '''
    Retorna el costo acumulado de comenzar en el nodo 'node' y llegar al nodo 'dest_node' siguiendo la política 'policy'
    '''
    # Obtener el camino más corto desde el nodo de inicio al nodo de destino dada la política
    path = [node]
    while node != dest_node:
        node = policy[node]
        path.append(node)
    # Calcular el costo acumulado del camino más corto
    cost = 0
    for index in range(len(path)-1):
        cost += get_edge_cost(grafo, path[index], path[index+1])
    return cost

class SSPEnv:
    '''
    Entorno de aprendizaje para el problema de encontrar el camino más corto en un grafo
    '''
    def __init__(self, grafo, start_state, terminal_state):
        '''Constructor de la clase	
        # Parámetros:
            grafo (Grafo): Grafo con el que se va a trabajar
        '''
        assert grafo is not None, 'El grafo no puede ser None'
        self.grafo = grafo
        self.num_nodos = grafo.number_of_nodes()
        assert start_state is not None, 'El estado inicial no puede ser None'
        self.__start_state = start_state
        self.__current_state = self.__start_state
        assert terminal_state is not None, 'El estado terminal no puede ser None'
        self.__terminal_state = terminal_state
        self.__num_states = grafo.number_of_nodes()
        self.__num_actions = {k : v for k, v in map(lambda item: (item[0], len(item[1])), nx.to_dict_of_lists(grafo).items())} # Diccionario con el número de acciones por estado
        self.__adjacency_dict_of_lists = nx.to_dict_of_lists(self.grafo)
        self.__q_table = self.dict_states_actions_zeros()

    def __str__(self):
        return f'EnvShortestPath {self.grafo}'
    @property
    def num_states(self):
        return self.__num_states
    @property
    def num_actions(self):
        return self.__num_actions
    @property
    def start_state(self):
        return self.__start_state
    @property
    def terminal_state(self):
        return self.__terminal_state
    @property
    def current_state(self):
        return self.__current_state

    def dict_states_zeros(self):
        ''' Retorna un diccionario con los estados con valor 0. Es útil para inicializar la tabla del número de visitas a cada estado, por ejemplo. Tiene la forma {estado: 0, ..., estado: 0}
        '''
        return { state: 0 for state, actions in nx.to_dict_of_lists(self.grafo).items()}

    def dict_states_actions_zeros(self):
        ''' Retorna un diccionario con los estados y acciones con valor 0. Es útil para inicializar la tabla Q, o la tabla de la cantidad de veces que se ha visitado cada par estado-acción. Tiene la forma {estado: {accion: 0, ..., accion: 0}, ..., estado: {accion: 0, ..., accion: 0}}
        ''' 
        G = self.grafo
        return {state: {action : 0 for action in actions} for state, actions in nx.to_dict_of_lists(G).items()}
    
    def dict_states_actions_random(self):
        ''' Retorna un diccionario con los estados y acciones con valor aleatorio. Es útil para inicializar la tabla Q, o la tabla de la cantidad de veces que se ha visitado cada par estado-acción. Tiene la forma {estado: {accion: valor_aleatorio, ..., accion: valor_aleatorio}, ..., estado: {accion: valor_aleatorio, ..., accion: valor_aleatorio}}
        ''' 
        G = self.grafo
        return {state: {action : np.random.random() for action in actions} for state, actions in nx.to_dict_of_lists(G).items()}

    def reset(self):
        '''Reiniciar el entorno de aprendizaje'''
        self.__current_state = self.__start_state
        print(f'Entorno reiniciado. Grafo con {self.num_nodos} nodos. Estado inicial: {self.__start_state}. Estado final: {self.__terminal_state}')
    
    def check_state(self, state):
        '''### Verificar si un estado es válido y/o terminal
        Parámetros:
            state (int): Estado a verificar
        Retorno:
            valid (bool): Indica si el estado es terminal y arroja un error si no es válido
        '''
        assert state in self.__adjacency_dict_of_lists.keys(), f'El estado {state} no está en el grafo'
        return state == self.terminal_state
    
    def take_action(self, state, action, distribution='expectation-lognormal'):
        '''### Tomar una acción en el entorno de aprendizaje
        Parámetros:
            state (int): Estado actual
            action (int): Acción a tomar
        Retorno:
            next_state (int): Siguiente estado
            reward (float): Recompensa (o costo) por tomar la acción
            terminated (bool): Indica si el episodio terminó
        '''
        # Obtener los arcos del nodo actual (estado actual)
        assert self.grafo.has_edge(state, action), f'La acción {action} no está en los arcos del nodo {state}'
        next_state = action
        cost = get_edge_cost(self.grafo, state, next_state, distribution)
        terminated = self.check_state(next_state)
        reward = - cost
        self.__current_state = next_state
        info = {'estado':state, 'recompensa': reward, 'terminado': terminated}
        return next_state, reward, terminated, info
    
    def calculate_optimal_qtable(self):
        '''### Calcular la tabla Q óptima
        '''
        pass