import os
import pickle
import datetime as dt


def load_graph(file_name, base_dir, folder_name="networks"):
    # Carpeta donde se cargará el archivo debe estar en el mismo directorio que este script.
    file_path = os.path.join(base_dir, folder_name, file_name)
    print(file_path)
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            grafo = pickle.load(f)
        return grafo
    else:
        print("El archivo no existe")
        return None


def save_graph(Object, grafo_perceptron, base_dir, folder_name="networks"):
    """Guarda el grafo en un archivo pickle en la carpeta networks (si no existe, se crea)

    Args:
        Object (PerceptronApp): Objeto de la clase PerceptronApp
        grafo_perceptron (nx.DiGraph): Grafo del perceptron
        base_dir (str): Directorio base donde se guardará la carpeta networks
        folder_name (str, optional): Nombre de la carpeta donde se guardarán los archivos. Por defecto es "networks".

    Returns:
        bool: True si el archivo fue guardado exitosamente, False en caso contrario
    """
    # Carpeta donde se guardará el archivo
    folder_path = os.path.join(base_dir, folder_name)
    # print(Object.capas, Object.nodos_por_capa)
    name = f'{str(grafo_perceptron)}_CreatedID {dt.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")}'
    # Generar un nombre único para el archivo
    nombre_archivo = f"{str(name)}.pkl"

    try:
        # Crear la carpeta si no existe
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Guardar el grafo en un archivo pickle
        file_path = os.path.join(folder_path, nombre_archivo)
        with open(file_path, "wb") as f:
            pickle.dump(grafo_perceptron, f)

        # Imprimir mensaje de éxito
        # print(f"Grafo guardado en {file_path}")

        return True
    except Exception as e:
        # Manejar cualquier error que pueda ocurrir durante el proceso de escritura
        print("Error al guardar el grafo:", e)
        return False


def get_folders_list(base_dir, folder_name="results"):
    """Obtiene la lista de nombres de carpetas existentes en la carpeta folder_name. Por defecto, folder_name es "results" """
    folder_path = os.path.join(base_dir, folder_name)
    # Listar los archivos en la carpeta networks filtrando por los archivos .pkl
    folder_list = [
        folder
        for folder in os.listdir(folder_path)
        if os.path.isdir(os.path.join(folder_path, folder))
    ]
    return folder_list


def get_pkl_files_in_folder(base_dir, folder_name="results"):
    """Obtiene la lista de archivos .pkl en la carpeta folder_name. Por defecto, folder_name es "results" """
    folder_path = os.path.join(base_dir, folder_name)
    # Listar los archivos en la carpeta networks filtrando por los archivos .pkl
    file_list = [file for file in os.listdir(folder_path) if file.endswith(".pkl")]
    return file_list
