import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))  # noqa: E402
import pickle
import osmnx as ox
import json
import inspect
from datetime import datetime
from RLib.agents.ssp import QAgentSSP
from RLib.utils.dijkstra import get_q_table_for_path
from RLib.utils.serializers import serialize_table

# ======================= Save and Load =======================


def save_model_results(agent: QAgentSSP, results_path: bool = None):
    """
    Guarda el agent en un archivo usando pickle, actualiza el storage_path del agent. Si la carpeta results_path no existe, la crea y guarda el archivo en ella.
    Además, guarda la tabla

    Parameters
    ----------
    agent: object
        agent a guardar
    results_path: str
        ruta de la carpeta donde se guardará el archivo (por defecto es "results")
    """
    # Verificar si results_path es None, y si es así, establecer la ruta del archivo que llama a la función
    if results_path is None:
        caller_file = inspect.getfile(inspect.currentframe().f_back)
        # Carpeta del archivo que llama a la función
        results_path = os.path.dirname(caller_file)
        # Carpeta "results" en la carpeta del archivo que llama a la función
        results_path = os.path.join(results_path, "results")
        print(f"results_path: {results_path}")
    os.makedirs(results_path, exist_ok=True)
    # Obtener la fecha y hora actual
    fecha_hora_actual = datetime.now()
    # Formatear la fecha y hora como una cadena
    fecha_hora_str = fecha_hora_actual.strftime("%Y-%m-%d_%H-%M-%S")
    # Crear el nombre del archivo con la fecha y hora
    agent_filename = f"QAgentSSP_{agent.strategy}_{agent.num_episodes}_{agent.env.costs_distribution}_{fecha_hora_str}.pickle"
    q_table_filename = f"QTable_{agent.strategy}_{agent.num_episodes}_{agent.env.costs_distribution}_{fecha_hora_str}.json"
    q_table_sp_filename = f"QTableSP_{agent.strategy}_{agent.num_episodes}_{agent.env.costs_distribution}_{fecha_hora_str}.json"
    visits_state_action_filename = f"VisitsStateAction_{agent.strategy}_{agent.num_episodes}_{agent.env.costs_distribution}_{fecha_hora_str}.json"
    # Combinar el nombre del archivo con la ruta de la carpeta "results"
    agent_storage_path = os.path.join(results_path, agent_filename)
    q_table_storage_path = os.path.join(results_path, q_table_filename)
    q_table_sp_storage_path = os.path.join(results_path, q_table_sp_filename)
    visits_storage_path = os.path.join(
        results_path, visits_state_action_filename)
    # Obtener la tabla Q para el mejor camino
    shortest_path = agent.shortest_path
    q_table = agent.q_table
    q_table_sp = get_q_table_for_path(
        q_table, shortest_path)  # Q table for shortest path
    # Serializar la tabla Q para el mejor camino
    serialized_q_table = serialize_table(q_table)
    serialized_q_table_sp = serialize_table(q_table_sp)
    serialized_visits = serialize_table(agent.visits_actions)

    # Guardar el agent en el archivo usando pickle
    with open(agent_storage_path, "wb") as archivo:
        agent.storage_path = agent_storage_path
        pickle.dump(agent, archivo)
    # Guardar la tabla Q en el archivo JSON
    with open(q_table_storage_path, "w") as archivo:
        json.dump(serialized_q_table, archivo, indent=4)
    # Guardar la tabla Q para el mejor camino en el archivo JSON
    with open(q_table_sp_storage_path, "w") as archivo:
        json.dump(serialized_q_table_sp, archivo, indent=4)
    # Guardar la tabla de visitas en el archivo JSON
    with open(visits_storage_path, "w") as archivo:
        json.dump(serialized_visits, archivo, indent=4)


def load_model_results(nombre_archivo, ruta_carpeta="results"):
    """
    Carga el objeto desde un archivo usando pickle.

    Parameters
    ----------
    nombre_archivo: str
        nombre del archivo a cargar
    """
    # Combinar el nombre del archivo con la ruta de la carpeta "results"
    ruta_archivo = os.path.join(ruta_carpeta, nombre_archivo)

    try:
        # Cargar el objeto desde el archivo usando pickle
        with open(ruta_archivo, "rb") as archivo:
            objeto_cargado = pickle.load(archivo)
        return objeto_cargado
    except EOFError:
        print(
            f"Error: El archivo {ruta_archivo} está vacío o no contiene datos válidos.")
        return None


def find_files_by_keyword(keyword, ruta_carpeta):
    """
    Busca archivos en una carpeta que contengan una palabra clave y devuelve una lista con los nombres de los archivos encontrados.

    Parameters
    ----------
    keyword: str
        palabra clave a buscar en los nombres de los archivos
    ruta_carpeta: str
        ruta de la carpeta donde se buscarán los archivos

    Returns
    -------
    list
        lista con los nombres de los archivos que contienen la palabra clave
    """
    archivos_encontrados = [
        archivo
        for archivo in os.listdir(ruta_carpeta)
        if os.path.isfile(os.path.join(ruta_carpeta, archivo)) and keyword in archivo and archivo.endswith(".pickle")
    ]
    return archivos_encontrados


# ======================= Download graph =======================
def download_graph(
    north=-33.4283,
    south=-33.6298,
    east=-70.9051,
    west=-70.5099,
    filepath="data/santiago-penaflor",
    format="graphml",
):
    """
    Descarga los datos de OpenStreetMap y crea un grafo de la ciudad de Santiago de Chile.
    Si el archivo "santiago-penaflor.graphml" existe, carga el grafo desde el archivo.
    Si el archivo no existe, descarga los datos y crea el grafo, y luego lo guarda en el archivo.
    """
    filepath = f"{filepath}.{format}"
    # Verificar si el archivo existe
    if os.path.exists(filepath):
        # Si el archivo existe, cargar el grafo desde el archivo
        G = ox.load_graphml(filepath)
        print("Grafo cargado desde el archivo.")
    else:
        # Si el archivo no existe, descargar los datos y guardar el grafo
        print("Descargando datos y creando el grafo...")
        # coordenadas desde santiago a mi casa north: -33.3311, south: -33.6569, east: -70.5681, west: -70.7949
        G = ox.graph_from_bbox(
            north=north, south=south, east=east, west=west, network_type="drive"
        )
        # Guardar el grafo en un archivo
        # Obtener el grafo no dirigido de bbox si es que es dirigido # ensure graph is undirected (needed for shortest path routing)

        ox.save_graphml(G, filepath)
        print("Grafo descargado y guardado.")
    return G


def serialize_and_save_table(table, path, file_name):
    """	Serializa la tabla y la guarda en un archivo JSON en la ruta especificada.
    Parameters
    ----------
    table: dict
        tabla a serializar
    path: str
        ruta de la carpeta donde se guardará el archivo
    file_name: str
        nombre del archivo
    """

    if not os.path.exists(path):
        os.makedirs(path)
    serialized_table = serialize_table(table)
    json_table = json.dumps(serialized_table, indent=4)
    with open(os.path.join(path, file_name), "w") as f:
        f.write(json_table)
        f.close()
