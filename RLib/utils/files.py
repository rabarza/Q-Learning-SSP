import os
import pickle
from datetime import datetime
import osmnx as ox

# ======================= Save and Load =======================


def save_model_results(objeto, nombre="", path="results"):
    """
    Guarda el objeto en un archivo usando pickle.

    Parameters
    ----------
    objeto: object
        objeto a guardar
    name: str
        nombre del archivo a guardar (opcional)
    path: str
        ruta de la carpeta donde se guardará el archivo (por defecto es "results")
    """
    # Verificar si la carpeta "results" existe, y si no, crearla
    if not os.path.exists(path):
        os.makedirs(path)
    # Obtener la fecha y hora actual
    fecha_hora_actual = datetime.now()
    # Formatear la fecha y hora como una cadena
    fecha_hora_str = fecha_hora_actual.strftime("%Y-%m-%d_%H-%M-%S")
    # Crear el nombre del archivo con la fecha y hora
    nombre_archivo = f"{nombre}_{objeto.strategy}_{objeto.num_episodes}_{objeto.alpha}_{objeto.alpha}_{objeto.env.costs_distribution}_{fecha_hora_str}.pickle"

    # Combinar el nombre del archivo con la ruta de la carpeta "results"
    ruta_archivo = os.path.join(path, nombre_archivo)
    # Guardar el objeto en el archivo usando pickle
    with open(ruta_archivo, "wb") as archivo:
        pickle.dump(objeto, archivo)


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
        print(f"Error: El archivo {ruta_archivo} está vacío o no contiene datos válidos.")
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
        if os.path.isfile(os.path.join(ruta_carpeta, archivo)) and keyword in archivo
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
