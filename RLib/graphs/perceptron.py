import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))  # noqa: E402
import plotly.graph_objects as go
import networkx as nx
import random
from itertools import product
from RLib.utils.dijkstra import dijkstra_shortest_path
from typing import List


def create_perceptron_graph(nodes_by_layer: List[int] = [1, 1],
                            min_length: int = 1,
                            max_length: int = 20,
                            seed: int = 20) -> nx.DiGraph:
    """Crea un grafo dirigido que representa un perceptrón multicapa.

    Parameters
    ----------
    nodes_by_layer : List[int]
        Número de nodos en cada capa del perceptrón. El i-ésimo elemento de la lista indica el número de nodos en la i-ésima capa.
    min_length : int
        Longitud mínima de los arcos que conectan los nodos.
    max_length : int
        Longitud máxima de los arcos que conectan los nodos.
    seed : int
        Semilla para el generador de números aleatorios para garantizar reproducibilidad.

    Returns
    -------
    perceptron_graph : nx.DiGraph
        Grafo dirigido que representa el perceptrón multicapa.

    Raises
    ------
    TypeError
        Si los parámetros layers y nodes_by_layer no son listas.
    ValueError
        Si el número de nodos por capa es menor o igual a cero.
    
    """
    if not isinstance(nodes_by_layer, list) or not all(isinstance(n, int) for n in nodes_by_layer):
        raise TypeError("nodes_by_layer debe ser una lista de enteros.")
    if len(nodes_by_layer) < 2:
        raise ValueError("Debe haber al menos dos capas.")
    if not all(n > 0 for n in nodes_by_layer):
        raise ValueError(
            "El número de nodos por capa debe ser mayor que cero.")
    if not min_length > 0 and max_length > 0:
        raise ValueError("min_length y max_length deben ser positivos.")

    # Inicializar la semilla para los largos aleatorios
    random.seed(seed)

    # Crear un grafo dirigido para representar el perceptrón
    graph = nx.DiGraph()
    # Diccionario para almacenar las posiciones de los nodos
    positions = {}
    # Diccionario para mapear los nodos a sus identificadores
    x_step = 1.5  # Separación horizontal entre layers
    y_step = 1  # Separación vertical entre nodos en la misma capa
    node_counter = 1  # Contador de nodos

    for i, num_nodes in enumerate(nodes_by_layer):
        for j in range(num_nodes):
            node_name = node_counter
            positions[node_counter] = (
                i * x_step, - (j * y_step) + (nodes_by_layer[i] - 1) / 2)
            # Las posiciones se almacenan en la llave "pos" de cada nodo.
            graph.add_node(node_counter, pos=positions[node_counter])
            node_counter += 1

    # Conectar los nodos de acuerdo a la estructura del perceptrón.
    # Los nodos de la capa i se conectan con los nodos de la capa i+1
    for i in range(len(nodes_by_layer) - 1):
        current_layer = range(
            sum(nodes_by_layer[:i]) + 1, sum(nodes_by_layer[:i + 1]) + 1)
        next_layer = range(
            sum(nodes_by_layer[:i + 1]) + 1, sum(nodes_by_layer[:i + 2]) + 1)

        for src, tgt in product(current_layer, next_layer):
            graph.add_edge(src, tgt, length=random.randint(
                min_length, max_length))
    # Renombrar el último nodo como 0
    last_node = node_counter - 1
    nx.relabel_nodes(graph, {last_node: 0}, copy=False)

    return graph


def create_hard_perceptron_graph(nodes_by_layer: List[int] = [1, 1], min_length: int = 1, max_length: int = 20, costs_distribution: str = None, seed: int = 20) -> nx.DiGraph:
    """
    Crea un grafo dirigido que representa un perceptrón multicapa donde se eliminan los arcos que comienzan en un nodo que no está en el camino más corto y terminan en un nodo que sí está en el camino más corto.

    Parameters
    ----------
    nodes_by_layer : List[int]
        Número de nodos en cada capa del perceptrón. El i-ésimo elemento de la lista indica el número de nodos en la i-ésima capa.
    min_length : int
        Longitud mínima de los arcos que conectan los nodos.
    max_length : int
        Longitud máxima de los arcos que conectan los nodos.
    costs_distribution : str
        Distribución de los costos de los arcos. Puede ser 'uniform' o 'normal'.
    seed : int
        Semilla para el generador de números aleatorios para garantizar reproducibilidad.

    Returns
    -------
    perceptron_graph : nx.DiGraph
        Grafo dirigido que representa el perceptrón multicapa.

    Raises
    ------
    TypeError
        Si los parámetros layers y nodes_by_layer no son listas.
    """

    def remove_edges_to_shortest_path(graph, shortest_path):
        """Remover los arcos que comienzan en un nodo que no está en el camino más corto y terminan en un nodo que sí está en el camino más corto
        """
        edges_to_remove = []
        for node in graph.nodes:
            if node in shortest_path:
                continue
            # No remover los arcos del nodo inicial y final
            for neighbor in shortest_path[1:-1]:
                if graph.has_edge(node, neighbor):
                    edges_to_remove.append((node, neighbor))
                    break
        # Remover los arcos
        graph.remove_edges_from(edges_to_remove)

    graph = create_perceptron_graph(
        nodes_by_layer, min_length, max_length, seed)
    origin_node = 1
    target_node = 0
    _, _, shortest_path = dijkstra_shortest_path(
        graph, origin_node, target_node, distribution=costs_distribution)
    remove_edges_to_shortest_path(graph, shortest_path)
    return graph


def plot_network_graph(graph, use_annotations=True, label_pos=0.15):
    """
    Función para visualizar cualquier grafo dirigido con nodos y arcos, donde los arcos tienen una longitud `length`.

    Parameters
    ----------
    graph : nx.DiGraph
        Grafo dirigido que representa el perceptrón multicapa.
    use_annotations : bool, opcional
        Indica si se deben mostrar las etiquetas de los arcos.
    label_pos : float, opcional
        Posición de las etiquetas en los arcos.

    Returns
    -------
    None
        La función muestra la figura en el navegador.
    """

    # Crear un objeto figura
    fig = go.Figure()

    # Se agregan los arcos y luego los nodos a la figura.
    # De esta forma, los nodos se dibujan sobre los arcos.
    edge_x = []
    edge_y = []
    annotations = []  # Para almacenar las anotaciones de las etiquetas de los arcos
    hover_x = []
    hover_y = []
    hover_text = []
    segment_count = 10  # Número de segmentos en los que se divide cada arco

    for edge in graph.edges:
        # Omitir arcos recurrentes
        if edge[0] == edge[1]:
            continue
        x0, y0 = graph.nodes[edge[0]]['pos']  # Posición del nodo de origen
        x1, y1 = graph.nodes[edge[1]]['pos']  # Posición del nodo de destino
        edge_x.append(x0)
        edge_x.append(x1)
        edge_y.append(y0)
        edge_y.append(y1)
        # Agregar None para separar los arcos y evitar que se dibujen líneas entre ellos
        edge_x.append(None)
        edge_y.append(None)

        # Añadir longitud del arco al texto del arco
        # Obtener la longitud del arco
        length = graph.edges[edge].get('length', 0)

        if use_annotations:
            # Calcular el primer trozo (15%) del segmento para la anotación
            annot_x = x0 + (x1 - x0) * label_pos
            annot_y = y0 + (y1 - y0) * label_pos
            annotations.append(
                dict(
                    x=annot_x,
                    y=annot_y,
                    text=f'{length}',
                    showarrow=False,
                    font=dict(
                        size=12,
                        color="#888"
                    ),
                    align="center"
                )
            )
        else:
            # Calcular puntos intermedios para hover donde se muestra la longitud del arco
            for i in range(1, segment_count):
                # Combinacion convexa de los puntos de inicio y fin (i/segment_count) <= 1
                mid_x = x0 + (x1 - x0) * i / segment_count
                mid_y = y0 + (y1 - y0) * i / segment_count
                hover_x.append(mid_x)
                hover_y.append(mid_y)
                hover_text.append(f'Length: {length}')

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',  # Desactivar el hoverinfo en las líneas
        mode='lines'
    )
    # Crear traza de hover en los arcos si no se usan anotaciones
    if not use_annotations:
        hover_trace = go.Scatter(
            x=hover_x, y=hover_y,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                size=10,
                # Hacer los puntos transparentes
                color='rgba(255, 255, 255, 0)',
                line=dict(width=0)
            ),
            text=hover_text  # Añadir el texto de las etiquetas de los arcos
        )

    # Crear traza de nodos
    node_x = []
    node_y = []
    for node in graph.nodes:
        x, y = graph.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='Viridis',  # colorscale options
            # 'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            # 'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            # 'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            reversescale=True,
            color=[],
            size=30,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2
        )
    )

    # Asignar colores y texto a los nodos
    node_adjacencies = []
    node_text = []
    for node in graph.nodes:
        adjacencies = list(graph.neighbors(node))
        node_adjacencies.append(len(adjacencies))
        node_text.append(
            f'Node: {node}<br># of connections: {str(len(adjacencies))}')
    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text

    # Crear figura
    fig_data = [edge_trace, node_trace] + \
        ([hover_trace] if not use_annotations else [])
    fig = go.Figure(data=fig_data,
                    layout=go.Layout(
                        title='Perceptron graph<br>',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False,
                                   showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False,
                                   showticklabels=False),
                        annotations=annotations if use_annotations else []
                    ))

    # Mostrar la figura
    fig.show()


if __name__ == "__main__":
    # Crear un perceptrón con 3 capas y 2 nodos en cada capa
    perceptron_graph = create_perceptron_graph(nodes_by_layer=[1, 2, 10, 2, 1])
    plot_network_graph(perceptron_graph, use_annotations=True, label_pos=0.6)
    hard_perceptron_graph = create_hard_perceptron_graph(nodes_by_layer=[1, 3, 3, 3, 1],
                                                         costs_distribution='uniform')
    plot_network_graph(hard_perceptron_graph,
                       use_annotations=True, label_pos=0.6)
