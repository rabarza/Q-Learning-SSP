import json

def serialize_dict_of_dicts(q_star):
    serialized_q_star = {}
    for key, value in q_star.items():
        serialized_key = str(key)
        if isinstance(value, dict):
            serialized_value = {str(k): v for k, v in value.items()}
        elif isinstance(value, list):
            serialized_value = [v for v in value]
        else:
            raise TypeError(f"Tipo de valor no compatible: {type(value)}")
        serialized_q_star[serialized_key] = serialized_value
    return serialized_q_star


if __name__ == '__main__':
    q_star = {('Entrada', 0): [('Oculta 1', 0), ('Oculta 1', 1)], ('Oculta 1', 0): [('Oculta 2', 0), ('Oculta 2', 1), ('Oculta 2', 2), ('Oculta 2', 3)], ('Oculta 1', 1): [('Oculta 2', 0), ('Oculta 2', 1), ('Oculta 2', 2), ('Oculta 2', 3)], ('Oculta 2', 0): [('Salida', 0)], ('Oculta 2', 1): [('Salida', 0)], ('Oculta 2', 2): [('Salida', 0)], ('Oculta 2', 3): [('Salida', 0)], ('Salida', 0): []}
    # Serializar q_star
    serialized_q_star = serialize_dict_of_dicts(q_star)

    # Convertir a formato JSON
    json_q_star = json.dumps(serialized_q_star, indent=4)

    # Imprimir el resultado
    print(json_q_star)