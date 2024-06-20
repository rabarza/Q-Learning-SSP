    import sys
    import os

    # Añadir el directorio superior a RLib al PYTHONPATH
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

    # Verificar que se añadió correctamente
    print("PYTHONPATH:", sys.path)