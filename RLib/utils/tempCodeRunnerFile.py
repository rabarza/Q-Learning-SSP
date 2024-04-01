import sys
import os

# Agrega el directorio padre (q-learning_app) al PATH para permitir importaciones relativas
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import networkx as nx
from RLib.environments.ssp import SSPEnv


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
        return {
            "num_states": self.q_agent.num_states,
            "num_actions": self.q_agent.num_actions,
            "alpha": self.q_agent.alpha,
            "gamma": self.q_agent.gamma,
            "dynamic_alpha": self.q_agent.dynamic_alpha,
            "strategy": str(self.q_agent.strategy),
            "times_actions": self.q_agent.times_actions,
            "times_states": self.q_agent.times_states,
            "q_table": self.q_agent.q_table,
            "alpha_formula": self.q_agent.alpha_formula,
            "num_episodes": getattr(self.q_agent, "num_episodes", None),
            "policy": getattr(self.q_agent, "policy", None),
            "distribution": getattr(self.q_agent, "distribution", None),
        }





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
    serialized_q_star = serialize_dict_of_dicts(q_star)

    # Convertir a formato JSON
    json_q_star = json.dumps(serialized_q_star, indent=4)

    # Imprimir el resultado
    print(json_q_star)
