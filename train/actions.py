import sys

sys.path.append("../RLib")  # Agrega el directorio RLib al sys.path
sys.path.append("../city")  # Agrega el directorio city al sys.path

import os
import streamlit as st

from RLib.action_selection.action_selector import (
    EpsilonGreedyActionSelector,
    DynamicEpsilonGreedyActionSelector,
    UCB1ActionSelector,
    Exp3ActionSelector,
)

# from RLib.distributions.distributions import (
#     UniformDistribution,
#     NormalDistribution,
#     ExponentialDistribution,
#     LogNormalDistribution,
# )


def get_action_selector(selected_strategy):
    
    if selected_strategy == "e-greedy":
        epsilon = st.slider(
            "Epsilon", min_value=0.01, max_value=1.0, value=0.1, step=0.01
        )
        action_selector = EpsilonGreedyActionSelector(epsilon=epsilon)

    elif selected_strategy == "UCB1":
        st.latex(r"a_{t+1} = \argmax_{a'} Q_t(s,a') + c \sqrt{\frac{\log(T)}{N_t(s,a')}}")
        c = st.number_input('C', min_value=0.0, max_value=100.0, value=2.0, step=0.01)
        action_selector = UCB1ActionSelector(c=c)
        
    elif selected_strategy == "exp3 η dinámico":
        st.latex(r"a_{t+1} \sim P(a_{t+1} = a) = \frac{ \exp(\eta \cdot Q_{t}(s,a)) }{\sum_{a'}\exp(\eta \cdot Q_{t}(s,a'))}")

        select_dynamic_beta = st.selectbox(
            "Selecciona el parámetro η dinámico",
            ["t", "t / T", "sqrt(t)", "log(t+1)"],
        )
        action_selector = Exp3ActionSelector(beta=select_dynamic_beta)

    elif selected_strategy == "exp3 η constante":
        st.latex(r"a_{t+1} \sim P(a_{t+1} = a) = \frac{ \exp(\eta \cdot Q_{t}(s,a)) }{\sum_{a'}\exp(\eta \cdot Q_{t}(s,a'))}")

        fixed_beta = st.slider("η", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
        action_selector = Exp3ActionSelector(beta=fixed_beta)

    return action_selector

def get_cost_distribution():
    cost_distribution = st.selectbox(
        "Distribución de costos",
        ["uniforme", "normal", "exponencial", "lognormal"],
    )
    return cost_distribution