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


def get_action_selector(selected_strategy):
    if selected_strategy == "e-greedy":
        epsilon = st.slider(
            "Epsilon", min_value=0.01, max_value=1.0, value=0.1, step=0.01
        )
        action_selector = EpsilonGreedyActionSelector(epsilon=epsilon)

    elif selected_strategy == "UCB1":
        c = st.slider(
            "C", min_value=1.0, max_value=20.0, value=2.0, step=0.5
        )  # Asegúrar que todos los valores sean del mismo tipo (float).
        action_selector = UCB1ActionSelector(c=c)
    elif selected_strategy == "exp3 β dinámico":
        select_dynamic_beta = st.selectbox(
            "Selecciona el parámetro β dinámico",
            ["t", "t / T", "sqrt(t)", "log(t)"],
        )
        action_selector = Exp3ActionSelector(beta=select_dynamic_beta)

    elif selected_strategy == "exp3 β constante":
        fixed_beta = st.slider("β", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
        action_selector = Exp3ActionSelector(beta=fixed_beta)

    return action_selector
