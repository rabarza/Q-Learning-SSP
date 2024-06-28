from typing import List
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from itertools import cycle
import random
from RLib.agents.ssp import QAgentSSP
import sys
import os
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../..')))

# ================= Comparación de modelos =================

# ======================= Get label =======================


def get_label(agent):
    """
    Obtiene la etiqueta para el agente en el gráfico de comparación de modelos.

    Parameters
    ----------
    agent: QAgent
        Agente QAgent

    Returns
    -------
    label: str
        Etiqueta para el agente en el gráfico
    """
    label = {}
    # Adding the label by the strategy of the agent
    label["strategy"] = agent.strategy

    # Adding the label by the alpha of the agent
    if agent.dynamic_alpha:  # If learning rate depends on time
        label["alpha"] = f"α = {agent.alpha_formula}"
    else:
        label["alpha"] = f"α = {agent.alpha}"
    return (
        f"{agent.action_selector.get_label()} | {label['alpha']} | {label['strategy']}"
    )


# ======================= Matplotlib =======================
def plot_results_per_episode_comp(
    lista,
    criteria="avg score",
    compare_best=False,
    dpi=100,
    episodes=None,
    save=False,
    name_file="Results.png",
):
    """
    Realiza una comparación gráfica de la cantidad de pasos que tardó cada agente en un episodio en llegar a un estado terminal.

    Parameters
    ----------
    lista: list
        lista de objetos de la clase QLearningAgent
    criteria: str
        criterio de comparación entre los agentes. Puede ser 'steps', 'avg score' o 'avg q_values'
    dpi: int
        resolución de la imagen
    """
    plt.figure(dpi=dpi)
    # Lista de colores que deseas asignar a los gráficos
    colores = [
        "blue",
        "orange",
        "green",
        "red",
        "purple",
        "brown",
        "pink",
        "gray",
        "olive",
        "cyan",
    ]
    # Generador que cicla a través de la lista de colores infinitamente
    color_generator = cycle(colores)

    for model in lista:
        color = next(color_generator)
        episodes = model.num_episodes
        if criteria == "steps":
            values = model.steps
            values_best = model.steps
        elif criteria == "score":
            values = model.scores
            values_best = model.scores_best
        elif criteria == "avg score":
            values = model.avg_scores
            values_best = model.avg_scores_best
        elif criteria == "acum q_values":
            values = model.acum_q_values
            values_best = model.acum_q_values_best
        elif criteria == "avg q_values":
            values = model.avg_q_values
            values_best = model.avg_q_values_best
        else:
            raise ValueError("Invalid comparison criteria")

        label = get_label(model)
        # the parameter to add color in the plot in matplotlib: color = model
        plt.plot(range(episodes), values, label=label, color=color)
        (
            plt.plot(range(episodes), values_best, label=label, color=color)
            if compare_best
            else None
        )
    plt.xlabel("Episodes")
    plt.ylabel(criteria)
    plt.grid()
    plt.savefig(name_file) if save else None
    plt.show()


# ======================= Plotly =======================


# Función para generar colores aleatorios
def get_random_color():
    return "#" + "".join([random.choice("0123456789ABCDEF") for j in range(6)])


def group_by_keyword(lista, keyword):
    grupos = {}
    for agente in lista:
        # Obtener el valor de la keyword del agente
        clave = getattr(agente, keyword)
        if clave not in grupos:
            grupos[clave] = []
        grupos[clave].append(agente)
    return grupos.items()


def get_color_by_strategy(strategy: str):
    colors = {"e-greedy": "#FF0000", "UCB1": "#FF00FF", "exp3": "#00FF00"}

    if strategy in ["softmax"]:
        return colors["exp3"]
    elif strategy in ["exp3"]:
        return colors["exp3"]
    elif strategy in ["e-greedy", "e-decay", "e-truncated"]:
        return colors["e-greedy"]
    elif strategy in ["UCB1"]:
        return colors["UCB1"]
    else:
        return get_random_color()


def ajustar_intensidad_color(color, intensity_factor):
    # Extraer componentes RGB
    r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:], 16)

    # Ajustar la intensidad de cada componente
    r = min(max(int(r * intensity_factor), 0), 255)
    g = min(max(int(g * intensity_factor), 0), 255)
    b = min(max(int(b * intensity_factor), 0), 255)

    # Devolver el color ajustado en formato hexadecimal
    return f"#{r:02X}{g:02X}{b:02X}"


def plot_results_per_episode_comp_plotly(
    lista: List[QAgentSSP], criteria: str = "avg score", add_label: bool = True, compare_best: bool = False
):
    '''Genera un gráfico de comparación de resultados por episodio utilizando Plotly.

    Parameters
    ----------
    lista : list
        Lista de agentes QLearningAgent.
    criteria : str
        Criterio de comparación entre los agentes. Puede ser 'steps', 'score', 'avg score', 'error', 'policy error' o 'regret'.
    add_label : bool
        Indica si se añade una etiqueta a cada línea del gráfico.
    compare_best : bool
        Indica si se compara con los mejores resultados obtenidos por cada agente.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Gráfico de comparación de resultados por episodio.
    '''

    fig = go.Figure()

    criteria_mapping = {
        "steps": ("steps", "steps_best"),
        "score": ("scores", "scores_best"),
        "avg score": ("avg_scores", "avg_scores_best"),
        "error": ("max_norm_error", None),
        "policy error": ("max_norm_error_shortest_path", None),
        "regret": ("regret", None),
        "average regret": ("average_regret", None),
    }

    if criteria not in criteria_mapping:
        raise ValueError("Invalid comparison criteria")

    values_attr, values_best_attr = criteria_mapping[criteria]
    criteria_name = "Shortest Path Error" if criteria == "policy error" else criteria.capitalize()

    for estrategia, agentes in group_by_keyword(lista, "strategy"):
        try:
            color_base = get_color_by_strategy(estrategia)
        except KeyError:
            color_base = get_random_color()

        for idx, model in enumerate(agentes):
            # Ajustar la intensidad del color para cada línea dentro del grupo
            color_actual = ajustar_intensidad_color(color_base, 1 - 0.05 * idx)

            values = getattr(model, values_attr)
            values_best = getattr(
                model, values_best_attr) if values_best_attr and compare_best else None
            episodes = model.num_episodes

            label = get_label(model) if add_label else None

            if episodes > 600000:
                values = values[::10000]
                iterations = list(range(0, episodes, 10000))
            else:
                iterations = list(range(episodes))

            fig.add_trace(
                go.Scattergl(
                    x=iterations,
                    y=values,
                    mode="lines",
                    name=label,
                    line=dict(color=color_actual),
                )
            )

            if values_best is not None:
                values_best = values_best[::10]
                fig.add_trace(
                    go.Scattergl(
                        x=iterations,
                        y=values_best,
                        mode="lines",
                        name=label + " (Best)",
                        line=dict(color=color_actual, dash="dash"),
                    )
                )

    fig.update_layout(
        xaxis_title="Episodios",
        yaxis_title=criteria_name,
        title=dict(text="Comparación de Resultados por Episodio"),
    )

    return fig


def plot_results_per_episode_comp_matplotlib(
    lista: List[QAgentSSP], criteria: str = "avg score", add_label: bool = True, compare_best: bool = False, dpi: int = 150
):
    '''Genera un gráfico de comparación de resultados por episodio utilizando Matplotlib.

    Parameters
    ----------
    lista : list
        Lista de agentes QLearningAgent.
    criteria : str
        Criterio de comparación entre los agentes. Puede ser 'steps', 'score', 'avg score', 'error', 'policy error' o 'regret'.
    add_label : bool
        Indica si se añade una etiqueta a cada línea del gráfico.
    compare_best : bool
        Indica si se compara con los mejores resultados obtenidos por cada agente.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Gráfico de comparación de resultados por episodio.
    '''

    fig = plt.figure(dpi=dpi)
    ax = plt.gca()  # Obtener el eje actual

    criteria_mapping = {
        "steps": ("steps", "steps_best"),
        "score": ("scores", "scores_best"),
        "avg score": ("avg_scores", "avg_scores_best"),
        "error": ("max_norm_error", None),
        "policy error": ("max_norm_error_shortest_path", None),
        "regret": ("regret", None),
        "average regret": ("average_regret", None),
    }

    if criteria not in criteria_mapping:
        raise ValueError("Invalid comparison criteria")

    values_attr, values_best_attr = criteria_mapping[criteria]
    criteria_name = "Shortest Path Error" if criteria == "policy error" else criteria.capitalize()

    for estrategia, agentes in group_by_keyword(lista, "strategy"):
        try:
            color_base = get_color_by_strategy(estrategia)
        except KeyError:
            color_base = get_random_color()

        for idx, model in enumerate(agentes):
            # Ajustar la intensidad del color para cada línea dentro del grupo
            color_actual = ajustar_intensidad_color(color_base, 1 - 0.05 * idx)

            values = getattr(model, values_attr)
            values_best = getattr(
                model, values_best_attr) if values_best_attr and compare_best else None
            episodes = model.num_episodes

            label = get_label(model) if add_label else None

            if episodes > 600000:
                values = values[::10000]
                iterations = list(range(0, episodes, 10000))
            else:
                iterations = list(range(episodes))

            ax.plot(iterations, values, label=label, color=color_actual)

            if values_best is not None:
                values_best = values_best[::10]
                ax.plot(iterations, values_best, label=label +
                        " (Best)", color=color_actual, linestyle='--')

    ax.set_xlabel("Episodios")
    ax.set_ylabel(criteria_name)
    ax.set_title("Comparación de Resultados por Episodio")

    if add_label:
        ax.legend()

    plt.show()

    return fig
