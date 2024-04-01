import plotly.graph_objects as go
import networkx as nx


def plot_perceptron_graph(grafo_perceptron):  #
    #
    edge_x = []
    edge_y = []

    for edge in grafo_perceptron.edges():
        x0, y0 = grafo_perceptron.nodes[edge[0]]["pos"]
        x1, y1 = grafo_perceptron.nodes[edge[1]]["pos"]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    lengths = nx.get_edge_attributes(grafo_perceptron, "length")
    # print(f"{lengths=}")
    
    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color="#888"),
        hoverinfo="text",
        mode="lines",
        text=[f"Longitud: {lengths[edge]}" for edge in grafo_perceptron.edges()],
    )

    node_x = []
    node_y = []
    node_text = []
    node_adjacencies = []
    posiciones = nx.get_node_attributes(
        grafo_perceptron, "pos"
    )  # {('Entrada', 0): (0.0, 0.0), ('Oculta 1', 0): (1.5, -1.0), ('Salida', 0): (4.5, 0.0)}

    for node, adjacencies in zip(posiciones.keys(), grafo_perceptron.adjacency()):
        x, y = posiciones[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(
            f"Capa: {node[0]}<br>Nodo: {node[1]}<br># de salidas: {str(len(adjacencies[1]))}"
        )
        node_adjacencies.append(len(adjacencies[1]))

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hoverinfo="text",
        text=node_text,
        marker=dict(
            showscale=True,
            colorscale="Viridis",
            reversescale=True,
            color=node_adjacencies,
            size=30,
            colorbar=dict(
                thickness=15,
                title="Node Connections",
                xanchor="left",
                titleside="right",
            ),
            line_width=2,
        ),
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title="Perceptron graph<br>",
            titlefont_size=16,
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )

    return fig
