from typing import List, TypeVar, Dict
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from itertools import cycle
import random
from RLib.agents.ssp import QAgentSSP
import sys
import os
from sympy import sympify, factor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# ================= Comparación de modelos =================
CRITERIA_MAPPING = {
    "steps": ("steps", "steps_best"),
    "score": ("scores", "scores_best"),
    "reward": ("scores", "scores_best"),
    "average reward": ("avg_scores", "avg_scores_best"),
    "avg score": ("avg_scores", "avg_scores_best"),
    "error": ("max_norm_error", None),
    "shortest path error": ("max_norm_error_shortest_path", None),
    "regret": ("regret", None),
    "average regret": ("average_regret", None),
    "optimal paths": ("optimal_paths", None),
    "max_norm_error_normalized": ("max_norm_error_normalized", None),
    "max_norm_error_shortest_path_normalized": (
        "max_norm_error_shortest_path_normalized",
        None,
    ),
    "normalized error": ("max_norm_error_normalized", None),
    "normalized shortest path error": ("max_norm_error_shortest_path_normalized", None),
}

COLOR_MAPPING = {
    "e-greedy": "#FF0000",
    "e-decay": "$0000FF",
    "UCB1": "#D95319",
    "AsOpt-UCB": "#FF00FF",
    "Boltzmann": "#00FF00",
}

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
    label["alpha"] = f"α = {agent.alpha}"

    # If the agent has action selector
    if hasattr(agent, "action_selector") and agent.action_selector is not None:
        return f"{agent.action_selector.get_label()} | {label['alpha']} | {label['strategy']}"
    return f"{agent.get_label()} | {label['alpha']} | {label['strategy']}"


def preprocess_label(label: str) -> str:
    """
    Preprocesa una etiqueta dividiéndola por '|', limpiando espacios y reconcatenando las partes.

    Parameters
    ----------
    label : str
        Etiqueta original del agente.

    Returns
    -------
    str
        Etiqueta preprocesada y normalizada.
    """
    label_parts = label.split("|")
    # Limpiar espacios y concatenar las partes sin espacios
    cleaned_parts = [part.strip() for part in label_parts]
    # Concatenar las partes con un guion bajo
    concatenated_label = "|".join(cleaned_parts).replace(" ", "")
    return concatenated_label


# Función para preprocesar alpha, eliminar espacios y normalizar la expresión
def preprocess_alpha(alpha: str) -> str:
    # Eliminar espacios en la fórmula
    alpha = alpha.replace(" ", "")

    # Reemplazar 'N(s,a)' por 'N_sa' y 'N(s)' por 'N_s'
    alpha = alpha.replace("N(s,a)", "N_sa")
    alpha = alpha.replace("N(s)", "N_s")

    # Convertir la fórmula a una expresión simbólica con sympify
    expr = sympify(alpha)

    # Factorizar la expresión para normalizarla
    factored_expr = factor(expr)

    # Convertir la expresión factorizada de vuelta a string
    return str(factored_expr)


T = TypeVar(
    "T"
)  # Esto permite que la función sea genérica y acepte cualquier tipo de objeto


def group_by_attribute(objects_list: List[T], keyword: str) -> Dict[str, List[T]]:
    """Agrupa los elementos de una lista de objetos por el valor de un atributo. Es útil para agrupar agentes por estrategia.

    - agrupa los agentes por la estrategia que utilizan e.g.::

        group_by_attribute(agentes, "strategy")

    Parameters
    ----------
    objects_list : List[object]
        objects_list de objetos a agrupar.

    keyword : str
        Atributo por el cual se agruparán los objetos.

    Returns
    -------
    grupos : dict
        Diccionario con los grupos de objetos en listas. Las claves son los valores únicos del atributo y los valores son listas de objetos.

    Examples
    --------
    >>> agentes = [agente1, agente2, agente3]
    >>> grupos = group_by_attribute(agentes, "strategy")
    >>> for estrategia, agentes in grupos:
    >>>     print(f"Agentes con estrategia {estrategia}: {agentes}")
    """

    grouped_lists_dict = {}
    sorted_list = sorted(objects_list, key=lambda x: getattr(x, keyword))
    for obj in sorted_list:
        key = str(getattr(obj, keyword))  # igual a obj.keyword
        grouped_lists_dict.setdefault(key, []).append(obj)
    return grouped_lists_dict


def get_random_color():
    return "#" + "".join([random.choice("0123456789ABCDEF") for j in range(6)])


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
    agents_list: List[QAgentSSP], criteria: str = "error", add_label: bool = True
):
    """
    Genera un gráfico de comparación de resultados por episodio utilizando Plotly y crea un DataFrame
    de los resultados de cada agente.

    Parameters
    ----------
    agents_list : list
        Lista de agentes QLearningAgent.
    criteria : str
        Criterio de comparación entre los agentes. Puede ser 'steps', 'score', 'avg score', 'error', 'policy error' o 'regret'.
    add_label : bool
        Indica si se añade una etiqueta a cada línea del gráfico.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Gráfico de comparación de resultados por episodio.
    """
    fig = go.Figure()

    # Obtener lista de valores únicos de alpha
    unique_alphas = sorted(set(agent.alpha for agent in agents_list))

    # Usar un diccionario con las fórmulas agrupadas por su clave preprocesada
    grouped_alphas = {preprocess_alpha(alpha): alpha for alpha in unique_alphas}

    if criteria not in CRITERIA_MAPPING:
        raise ValueError(f"Invalid comparison criteria: {criteria}")

    values_attr, _ = CRITERIA_MAPPING[criteria]
    criteria_name = criteria.capitalize()

    # Para controlar qué etiquetas ya se han mostrado en la leyenda
    shown_labels = set()

    for estrategia, agentes in group_by_attribute(agents_list, "strategy").items():
        color_base = COLOR_MAPPING.get(estrategia, get_random_color())
        for idx, agent in enumerate(agentes):
            # Ajustar la intensidad del color para cada línea dentro del grupo
            color_actual = ajustar_intensidad_color(color_base, 1 - 0.0005 * idx)
            # Cargar los resultados del agente
            data = agent.results()

            if not data or values_attr not in data:
                print(f"Attribute {values_attr} not found for agent {agent}. Skipping.")
                continue

            values = data[values_attr]
            episodes = agent.num_episodes

            # Crear DataFrame para este agente
            agent_df = pd.DataFrame(
                {"episode": list(range(episodes)), criteria: values}
            )

            # Submuestrear para reducir la cantidad de puntos en el gráfico
            n = 25
            agent_df = agent_df.iloc[::n, :]

            # Obtener la etiqueta del agente
            original_label = get_label(agent) if add_label else f"Agent {agent.id}"
            preprocessed_label = preprocess_label(original_label)
            # Determinar si mostrar la leyenda para este trace
            if preprocessed_label not in shown_labels:
                show_legend = True
                shown_labels.add(preprocessed_label)
                legend_name = original_label  # Nombre en la leyenda
            else:
                show_legend = False
                legend_name = None  # No mostrar en la leyenda
                
            # Obtener el alpha del agente y preprocesarlo
            normalized_alpha = preprocess_alpha(agent.alpha)
            
            # Agregar la curva del agente al gráfico
            fig.add_trace(
                go.Scattergl(
                    x=agent_df["episode"],
                    y=agent_df[criteria],
                    mode="lines",
                    name=legend_name,
                    legendgroup=preprocessed_label,
                    showlegend=show_legend,
                    line=dict(color=color_actual),
                    hovertext=original_label,
                    visible=True,
                    # se agrega información adicional para poder filtrar por alpha
                    customdata=[normalized_alpha]
                    * len(agent_df),
                )
            )

    # Crear botones para filtrar por la fórmula de alpha (usando las claves normalizadas)
    buttons = []
    for normalized_alpha in grouped_alphas.keys():
        # La visibilidad de las curvas corresponde al alpha preprocesado. 
        # Si el alpha del agente coincide con el alpha normalizado, la curva es visible para ese botón.
        visible = [trace["customdata"][0] == normalized_alpha for trace in fig.data]
        buttons.append(
            dict(
                method="update",
                label=f"α: {grouped_alphas[normalized_alpha]}",  # Usar la fórmula original como etiqueta
                args=[
                    {"visible": visible},
                    {"title": f"{criteria} - α: {grouped_alphas[normalized_alpha]}"},
                ],
            )
        )

    # El primer botón muestra todas las curvas (todas las alphas)
    buttons.insert(
        0,
        dict(
            label="Mostrar todos",
            method="update",
            args=[
                {"visible": [True] * len(fig.data)},
                {"title": f"{criteria} - Todos los α"},
            ],
        ),
    )

    # Agrupar los botones en un menú desplegable en el layout
    fig.update_layout(
        updatemenus=[
            dict(
                type="dropdown",  # 'dropdown' (lista de opciones)
                direction="down",
                buttons=buttons,
                active=0,
                x=0,
                xanchor="left",
                y=1,
                yanchor="top",
            )
        ],
        xaxis_title="Episodios",
        yaxis_title=criteria_name,
        title="Comparación de Resultados por Episodio",
        showlegend=True,
        legend_title="Agentes",
        legend=dict(
            orientation="v",
            x=1.02,
            xanchor="left",
            y=1,
            yanchor="top",
            font=dict(
                size=10
            ),  # Reduce el tamaño de la fuente para disminuir el espacio entre entradas
            tracegroupgap=5,  # Reduce el espacio entre grupos de trazas
        ),
        hoverlabel=dict(font_size=16),
    )

    return fig


def plot_results_per_episode_comp_matplotlib(
    lista: List[QAgentSSP],
    criteria: str = "nomalized shortest path error",
    add_label: bool = True,
    dpi: int = 150,
):
    """Genera un gráfico de comparación de resultados por episodio utilizando Matplotlib.

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
    """

    fig = plt.figure(dpi=dpi)
    ax = plt.gca()  # Obtener el eje actual

    if criteria not in CRITERIA_MAPPING:
        raise ValueError("Invalid comparison criteria")

    values_attr, values_best_attr = CRITERIA_MAPPING[criteria]
    criteria_name = (
        "Shortest Path Error" if criteria == "policy error" else criteria.capitalize()
    )

    for estrategia, agentes in group_by_attribute(lista, "strategy"):
        color_base = COLOR_MAPPING.get(estrategia, get_random_color())

        for idx, agent in enumerate(agentes):
            # Ajustar la intensidad del color para cada línea dentro del grupo
            color_actual = ajustar_intensidad_color(color_base, 1 - 0.05 * idx)

            values = agent.results()[values_attr]
            episodes = agent.num_episodes

            label = get_label(agent) if add_label else None

            if episodes > 600000:
                values = values[::10000]
                iterations = list(range(0, episodes, 10000))
            else:
                iterations = list(range(episodes))

            ax.plot(iterations, values, label=label, color=color_actual)

    ax.set_xlabel("Episodios")
    ax.set_ylabel(criteria_name)
    ax.set_title("Comparación de Resultados por Episodio")

    if add_label:
        ax.legend()

    plt.show()

    return fig
