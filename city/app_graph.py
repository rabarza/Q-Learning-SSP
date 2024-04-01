import sys
sys.path.append('../RLib')  # Agrega el directorio RLib al sys.path

import streamlit as st
import osmnx as ox
import networkx as nx
import pydeck as pdk



QUERIE_PARAMS = {
    'Piedmont, California': {
        'query': 'Piedmont, California',	
        'network_type': 'drive',
    },
    'Santiago, Chile': {
        'query': 'Santiago, Chile',
        'network_type': 'drive',
    },
    'Peñaflor, Chile': {
        'query': 'Peñaflor, Chile',
        'network_type': 'drive',
    }
}

@st.cache_resource
def download_graph(parameters:dict, query:str='Piedmont, California'):
    if query == 'Piedmont, California':
        G = ox.graph_from_place(**parameters[query])
    if query == 'Santiago, Chile':
        G = ox.graph_from_place(**parameters[query])
    else:
        G = ox.graph_from_place(**parameters[query])
    G = ox.utils_graph.get_largest_component(G, strongly=True)
    return G
    
def extract_edge_coordinates(row):
    return list(zip(row.geometry.xy[0], row.geometry.xy[1]))


class CityGraph:
    def __init__(self, parameters:dict, query:str='Piedmont, California'):
        
        self.parameters = parameters
        self.query = query
        self.G = download_graph(parameters=self.parameters, query=self.query)
        self.gdf_nodes, self.gdf_edges = ox.graph_to_gdfs(self.G)
        self.gdf_nodes['osmid'] = self.gdf_nodes.index
        self.gdf_nodes['neighbors'] = self.gdf_nodes.apply(lambda row: len(list(self.G.neighbors(row.name))), axis=1)
        self.gdf_edges['coordinates'] = self.gdf_edges.apply(extract_edge_coordinates, axis=1)
        self.view_state = pdk.ViewState(
            latitude=self.gdf_nodes.geometry.y.mean(), 
            longitude=self.gdf_nodes.geometry.x.mean(), 
            zoom=13.5
        )
        self.edges_layer = pdk.Layer(
            'PathLayer',  # Cambiar a PathLayer para manejar múltiples coordenadas
            self.gdf_edges,
            get_path='coordinates',  # Usar la columna 'coordinates'
            get_width=4,
            get_color=[255, 0, 0, 180],
            pickable=False
        )
        self.nodes_layer = pdk.Layer(
            "ScatterplotLayer",
            data= self.gdf_nodes,
            get_position=["x", "y"],
            get_color=[0, 128, 255, 160],
            get_radius=20,
            pickable=True
        )
        self.tooltip = {
            "html": "<b>ID del Nodo:</b> {osmid} <br> <b>Coordenadas:</b> {x}, {y} <br> <b>Vecinos:</b> {neighbors}",
            "style": {
                "backgroundColor": "steelblue",
                "color": "white"
            }
            
        }
        self.deck = pdk.Deck(
            layers=[self.nodes_layer, self.edges_layer], 
            initial_view_state=self.view_state,
            tooltip=self.tooltip
        )
        
    def show(self):
        st.pydeck_chart(self.deck)
        
    def add_node_layer(self, id_node, color, radius=20):
        # Encuentra las coordenadas del nodo específico
        coords_node = self.gdf_nodes.loc[id_node, ['x', 'y']].values.tolist()
        # Crear una capa para resaltar el nodo específico
        layer = pdk.Layer(
            'ScatterplotLayer',
            data=[{'coordinates': coords_node}],
            get_position='coordinates',
            get_color=color,
            get_radius=radius,
        )
        self.deck.layers.append(layer)
        
    def add_edge_layer(self, id_edge, color, width=4):
        
        # Encuentra las coordenadas del nodo específico
        coords_edge = self.gdf_edges.loc[id_edge, 'coordinates']
        # Crear una capa para resaltar el nodo específico
        layer = pdk.Layer(
            'PathLayer',
            data=[{'coordinates': coords_edge}],
            get_path='coordinates',
            get_color=color,
            get_width=width,
        )
        self.deck.layers.append(layer)
        
    def add_edges_layer(self, id_edges, color, width=4):
        
        # Encuentra las coordenadas del nodo específico
        coords_edges = self.gdf_edges.loc[id_edges, 'coordinates']
        # Crear una capa para resaltar el nodo específico
        layer = pdk.Layer(
            'PathLayer',
            data=[{'coordinates': coords_edges}],
            get_path='coordinates',
            get_color=color,
            get_width=width,
        )
        self.deck.layers.append(layer)
        
    def add_nodes_layer(self, id_nodes, color, radius=20):
                
        # Encuentra las coordenadas del nodo específico
        coords_nodes = self.gdf_nodes.loc[id_nodes, ['x', 'y']].values.tolist()
        # Crear una capa para resaltar el nodo específico
        layer = pdk.Layer(
            'ScatterplotLayer',
            data=[{'coordinates': coords_nodes}],
            get_position='coordinates',
            get_color=color,
            get_radius=radius,
        )
        self.deck.layers.append(layer)