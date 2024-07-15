import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# Función para plotear los regrets
def plot_bandits_regret(bandits, criteria="regret"):
    fig = go.Figure()
    
    # Definir un diccionario de colores para cada estrategia
    strategy_colors = {
        'UCB': 'blue',
        'e-greedy': 'green',
        'EXP3': 'red',
        'Boltzmann': 'orange'
    }

    for bandit in bandits:
        episodes = len(bandit.regret_history)
        iterations = list(range(episodes))
        selected_attribute = {"regret": "regret_history", "average regret": "average_regret_history", "pseudo regret": "pseudo_regret_history", "rewards": "rewards", "pulls": "arm_pulls"}
        values = getattr(bandit, selected_attribute[criteria])          
            
        label = f'Bandit {bandit.strategy}'
        # Asignar un color diferente a cada bandit dependiendo de su estrategia (bandit.strategy)
        color = strategy_colors.get(bandit.strategy, 'black')  # Color por defecto negro si la estrategia no está en el diccionario

        if criteria != "pulls":
            fig.add_trace(
                go.Scattergl(
                    x=iterations,
                    y=values,
                    mode="lines",
                    name=label,
                    line=dict(color=color),
                )
            )
        else:
            fig.add_trace(
                go.Bar(
                    x=list(map(lambda x: f"Arm {x}", range(len(values)))),
                    y=values,
                    name=label,
                    marker=dict(color=color),
                )
            )
    if criteria != "pulls":
        fig.update_layout(
            xaxis_title="Episodios",
            yaxis_title=criteria.capitalize(),
            title=dict(text="Comparación de Regret por Episodio"),
        )
    else:
        fig.update_layout(
            xaxis_title="Episodios",
            yaxis_title=criteria.capitalize(),
            title=dict(text="Comparación de Regret por Episodio"),
        )

    return fig
