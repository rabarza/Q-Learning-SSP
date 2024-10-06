import networkx as nx
from typing import List, Dict, Any

def ensure_proper_policy(graph: nx.DiGraph, target_node: Any) -> nx.DiGraph:
    """Asegura que desde cualquier nodo haya un camino al nodo destino. Retorna un subgrafo a partir del grafo original."""
    # Invertir el grafo para hacer búsqueda inversa desde el nodo objetivo
    reversed_graph = graph.reverse() # The reverse is a graph with the same nodes and edges but with the directions of the edges reversed.
    
    # Encuentra todos los nodos que pueden llegar al nodo objetivo
    reachable_from_target = nx.single_source_shortest_path_length(reversed_graph, target_node)
    
    # Filtra y crea un subgrafo con solo los nodos que pueden llegar al objetivo
    nodes_to_keep = set(reachable_from_target.keys())
    subgraph = graph.subgraph(nodes_to_keep).copy()
    
    return subgraph