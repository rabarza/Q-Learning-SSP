import osmnx as ox
import networkx as nx
import pandas as pd
import numpy as np


# Descargar el grafo y obtener el componente conexo más grande
location_name = "Piedmont, California"
G = ox.graph_from_place(location_name, network_type="drive")
# Añadir el atributo de velocidad a las aristas (speed_kph)
G = ox.add_edge_speeds(G)
G.to_directed()
G = ox.utils_graph.get_largest_component(
    G, strongly=True
)
# fig, ax = ox.plot_graph(G, node_size=20, node_color="red", edge_linewidth=0.5, figsize=(10,10), save=True, filepath="piedmont.svg")

df = ox.graph_to_gdfs(G, nodes=False)
