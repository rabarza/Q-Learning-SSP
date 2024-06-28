import json
import networkx as nx
from .table_utils import resta_diccionarios


def serialize_table(table):
    """Serializa un diccionario de diccionarios, diccionario listas o de valores numéricos a un formato compatible con JSON.
    Args:
        table (dict): Diccionario de diccionarios, listas o valores numéricos a serializar.
    Returns:
        dict: Diccionario serializado.
    """
    if not table:
        return {}
    serialized_table = {}
    for key, value in table.items():
        serialized_key = str(key)
        if isinstance(value, dict):
            serialized_value = {str(k): v for k, v in value.items()}
        elif isinstance(value, list):
            serialized_value = [v for v in value]
        elif isinstance(value, int) or isinstance(value, float):
            serialized_value = value
        elif isinstance(value, tuple):
            serialized_value = str(value)
        elif value is None:
            serialized_value = None
        else:
            raise TypeError(f"Tipo de valor no compatible: {type(value)}")
        serialized_table[serialized_key] = serialized_value
    return serialized_table


class SSPEnvSerializer:
    def __init__(self, ssp_env):
        self.ssp_env = ssp_env

    def to_dict(self):
        # Serializar el grafo a un formato compatible con JSON
        serialized_graph = nx.node_link_data(self.ssp_env.grafo)
        return {
            "num_states": self.ssp_env.num_states,
            "num_actions": self.ssp_env.num_actions,
            "start_state": self.ssp_env.start_state,
            "terminal_state": self.ssp_env.terminal_state,
            "graph": serialized_graph,  # Grafo serializado
        }


class QAgentSSPSerializer:
    def __init__(self, q_agent):
        self.q_agent = q_agent

    def to_dict(self):
        error_q_table = serialize_table(
            resta_diccionarios(self.q_agent.q_star, self.q_agent.q_table)
        )
        shortest_path = list(
            map(lambda x: str(x), getattr(self.q_agent, "shortest_path", None))
        )
        optimal_policy = serialize_table(getattr(self.q_agent, "optimal_policy", {}))
        return {
            "strategy": str(self.q_agent.strategy),
            "num_episodes": getattr(self.q_agent, "num_episodes", None),
            "learning_rate": self.q_agent.alpha,
            "learning_rate_formula": self.q_agent.alpha_formula,
            "discount_rate": self.q_agent.gamma,
            "costs_distribution": getattr(self.q_agent, "distribution", None),
            "num_states": self.q_agent.num_states,
            "num_actions": serialize_table(self.q_agent.num_actions),
            "times_actions": serialize_table(self.q_agent.times_actions),
            "times_states": serialize_table(self.q_agent.times_states),
            "q_table": serialize_table(self.q_agent.q_table),
            "optimal_q_table": serialize_table(getattr(self.q_agent, "q_star", {})),
            "error_q_table": error_q_table,
            "shortest_path": shortest_path,
            "optimal_policy": optimal_policy,
            "max_norm_error": list(getattr(self.q_agent, "max_norm_error", [])),
            "max_norm_error_shortest_path": list(
                getattr(self.q_agent, "max_norm_error_shortest_path", [])
            ),
            "regret": list(getattr(self.q_agent, "regret", [])),
        }


if __name__ == "__main__":
    # Ejemplo de diccionario de diccionarios
    q_star = {
        ("Entrada", 0): {("Oculta 1", 0): 5},
        ("Oculta 1", 0): {
            ("Oculta 2", 0): 10,
            ("Oculta 2", 1): 14,
            ("Oculta 2", 2): 20,
        },
        ("Oculta 2", 0): {("Salida", 0): 15},
    }

    # Serializar q_star
    serialized_q_star = serialize_table(q_star)

    # Convertir a formato JSON
    json_q_star = json.dumps(serialized_q_star, indent=4)

    # Imprimir el resultado
    print(json_q_star)
    
    # Guardar en un archivo
    with open("q_star.json", "w") as f:
        f.write(json_q_star)
