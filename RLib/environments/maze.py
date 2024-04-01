import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

############ LABERINTO  ############
def generate_maze(rows, cols, start, end):
    # Inicializar todas las celdas como paredes (-100)
    maze = [[-100] * cols for _ in range(rows)]  

    # Definir punto de inicio
    maze[start[0]][start[1]] = -1

    # Utilizaremos un stack para rastrear las celdas visitadas
    stack = [start]  
    # Conjunto para almacenar las celdas visitadas
    visited = set([start]) # los conjuntos evitan duplicados
    
    # Mientras el stack no esté vacío
    while stack:
        current_row, current_col = stack[-1]

        # Obtener celdas vecinas no visitadas
        unvisited_neighbours = []
        if current_row > 1 and (current_row - 2, current_col) not in visited: # si la celda de arriba no ha sido visitada
            unvisited_neighbours.append((current_row - 2, current_col))
        if current_row < rows - 2 and (current_row + 2, current_col) not in visited: # si la celda de abajo no ha sido visitada
            unvisited_neighbours.append((current_row + 2, current_col))
        if current_col > 1 and (current_row, current_col - 2) not in visited: # si la celda de la izquierda no ha sido visitada
            unvisited_neighbours.append((current_row, current_col - 2))
        if current_col < cols - 2 and (current_row, current_col + 2) not in visited: # si la celda de la derecha no ha sido visitada
            unvisited_neighbours.append((current_row, current_col + 2))

        if unvisited_neighbours:
            # Elegir una celda vecina aleatoriamente
            next_row, next_col = random.choice(unvisited_neighbours)

            # Eliminar la pared entre la celda actual y la vecina
            maze[next_row][next_col] = -1
            maze[(current_row + next_row) // 2][(current_col + next_col) // 2] = -1 # celda intermedia entre la actual y la vecina

            visited.add((next_row, next_col))
            stack.append((next_row, next_col)) # añadir la celda vecina al stack

            if (next_row == end[0] and abs(next_col - end[1]) == 1) or (next_col == end[1] and abs(next_row - end[0]) == 1):
                # CORREGIR ESTO
                # Hay que derribar la pared entre la celda vecina y la meta 
                # Si la celda vecina es adyacente a la meta, terminar el algoritmo
                break

        else:
            # No hay celdas vecinas no visitadas, retroceder
            stack.pop() # eliminar la celda actual del stack
    # Definir meta
    maze[end[0]][end[1]] = 500
    return np.array(maze)

class Maze:
    def __init__(self, maze, start, goal):
        self.maze = maze
        self.start = start
        self.goal = goal
    
    def start_state(self):
        return self.index(self.start)
    
    def terminal_state(self, state, game):
        match game:
            case 'fire-walls':
                return state == self.index(self.goal)
            case 'pit-walls':
                row, col = self.position(state)
                return (self.maze[row, col] != -1)
            case _:
                raise ValueError("A valid game was expected")

    def take_action(self, state, action):
        row, col = self.position(state)

        if action == 0: # up
            row = max(row - 1, 0)
        elif action == 1: # down
            row = min(row + 1, self.maze.shape[0] - 1)
        elif action == 2: # left
            col = max(col - 1, 0)
        elif action == 3: # right
            col = min(col + 1, self.maze.shape[1] - 1)
        else:
            raise ValueError("Not a valid action")
        
        next_state = self.index((row, col))
        reward = self.maze[row, col]
        return next_state, reward
    
    def index(self, position):
        # índice = fila * tamaño_columnas + columna
        index = position[0] * self.maze.shape[1] + position[1]
        return index
    
    def position(self, index):
        # fila, columna = divmod(índice, tamaño_columnas)
        # fila = índice // tamaño_columnas
        # columna = índice % tamaño_columnas
        return divmod(index, self.maze.shape[1])
    
    def render(self, path = []):
        
        start_point, end_point = (self.start, self.goal)

        rows = self.maze.shape[0]
        cols = self.maze.shape[1]

        # Crear una figura y un eje
        fig, ax = plt.subplots()

        # Configurar el tamaño de la figura en función del tamaño del laberinto
        fig.set_size_inches(cols, rows)

        # Configurar límites del eje
        ax.set_xlim(0, cols)
        ax.set_ylim(rows, 0)

        # Ocultar ejes
        ax.set_axis_off()

        # Dibujar las paredes
        for row in range(rows):
            for col in range(cols):
                if self.maze[row, col] == -100:
                    rect = Rectangle((col, rows - row - 1), 1, 1, facecolor="black")
                    ax.add_patch(rect)

        # Dibujar el camino
        if path:
            for cell in path:
                path_rect = Rectangle((cell[1], rows - cell[0] - 1), 1, 1, facecolor="palegreen")
                ax.add_patch(path_rect)
                
        # Dibujar el punto de inicio
        start_row, start_col = start_point
        start_rect = Rectangle((start_col, rows - start_row - 1), 1, 1, facecolor="red")
        ax.add_patch(start_rect)

        # Dibujar la meta
        end_row, end_col = end_point
        end_rect = Rectangle((end_col, rows - end_row - 1), 1, 1, facecolor="lime")
        ax.add_patch(end_rect)

        # Mostrar la figura
        plt.show()
