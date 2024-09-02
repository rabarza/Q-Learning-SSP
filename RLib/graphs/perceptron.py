import plotly.graph_objects as go
import networkx as nx
import random


def create_perceptron_graph(nodes_by_layer=[1, 1],
                            min_length=1,
                            max_length=20,
                            seed=None) -> nx.DiGraph:
    """ 
    Crea un grafo dirigido que representa un perceptrón multicapa.

    Parámetros
    ----------
    nodes_by_layer : list of int
        Número de nodos en cada capa del perceptrón. El i-ésimo elemento de la lista indica el número de nodos en la i-ésima capa.
    min_length : int
        Longitud mínima de los arcos que conectan los nodos.
    max_length : int
        Longitud máxima de los arcos que conectan los nodos.
    seed : int, optional
        Semilla para el generador de números aleatorios para garantizar reproducibilidad.

    Retorna
    -------
    perceptron_graph : nx.DiGraph
        Grafo dirigido que representa el perceptrón multicapa.
    """
    if type(nodes_by_layer) != list:
        raise TypeError(
            "Los parámetros layers y nodes_by_layer deben ser listas.")

    # Inicializar la semilla para los números aleatorios
    if seed is not None:
        random.seed(seed)

    # Definir las capas del perceptrón y el número de nodos en cada capa
    layers = list(map(lambda x: f'Capa {x}', range(len(nodes_by_layer))))
    layers[0] = 'Entrada'
    layers[-1] = 'Salida'

    # Crear un grafo dirigido para representar el perceptrón
    perceptron_graph = nx.DiGraph()

    # Agregar nodos al grafo con las posiciones en el plano
    posiciones = {}
    x_step = 1.5  # Separación horizontal entre layers
    y_step = 1  # Separación vertical entre nodos en la misma capa

    for i, capa in enumerate(layers):
        for j in range(nodes_by_layer[i]):
            posiciones[(capa, j)] = (
                i * x_step, - (j * y_step) + (nodes_by_layer[i] - 1) / 2)
            # Agregar nodos al grafo con las posiciones definidas.
            # Las posiciones se almacenan en la llave "pos" de cada nodo.
            perceptron_graph.add_node((capa, j), pos=posiciones[(capa, j)])

    # Conectar los nodos de acuerdo a la estructura del perceptrón.
    # Los nodos de la capa i se conectan con los nodos de la capa i+1
    for i in range(len(layers) - 1):
        for j in range(nodes_by_layer[i]):
            for k in range(nodes_by_layer[i + 1]):
                random_length = random.randint(min_length, max_length)
                perceptron_graph.add_edge(
                    (layers[i], j),  # Nodo de origen (capa i, nodo j)
                    (layers[i + 1], k),  # Nodo de destino
                    length=random_length,  # Longitud del arco
                )

    return perceptron_graph


def plot_network_graph(graph, use_annotations=True, label_pos=0.15):
    """
    Función para visualizar cualquier grafo dirigido con nodos y arcos, donde los arcos tienen una longitud `length`.

    Parámetros
    ----------
    graph : nx.DiGraph
        Grafo dirigido que representa el perceptrón multicapa.
    use_annotations : bool
        Indica si se deben mostrar las etiquetas de los arcos.
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
