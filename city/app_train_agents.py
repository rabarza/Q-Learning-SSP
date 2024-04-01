import sys

sys.path.append("../RLib")  # Agrega el directorio RLib al sys.path
sys.path.append("../city")  # Agrega el directorio city al sys.path

import os
import streamlit as st
from city.app_graph import CityGraph, QUERIE_PARAMS

# from .app_results import city_selectbox
from time import sleep

from RLib.environments.ssp import SSPEnv
from RLib.agents.ssp import QAgentSSP
from RLib.utils.dijkstra_utils import get_optimal_policy, get_q_table_for_policy
from RLib.utils.file_utils import save_model_results
from RLib.action_selection.action_selector import (
    EpsilonGreedyActionSelector,
    DynamicEpsilonGreedyActionSelector,
    UCB1ActionSelector,
    Exp3ActionSelector,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

@st.cache_data(persist=True)
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def show():

    # Cargar el CSS
    load_css("styles/styles.css")
    # Seleccionar la ciudad y descargar el grafo
    if "city_select_train" in st.session_state:
        st.session_state.location_name = None
    location_name = st.sidebar.selectbox(
        "Seleccionar Ciudad", list(QUERIE_PARAMS.keys()), key=f"city_select_train"
    )

    G = CityGraph(QUERIE_PARAMS, query=location_name)

    # Visualizar en Streamlit
    st.markdown(
        f"<h1>Entorno Seleccionado: {location_name}</h1>", unsafe_allow_html=True
    )

    # Interfaz para seleccionar el nodo inicial y final
    all_nodes = list(G.G.nodes)
    with st.form("node_form"):
        default_node = (
            53035699 if location_name == "Piedmont, California" else all_nodes[-1]
        )
        orig_node = st.selectbox("Seleccionar Nodo Inicial", all_nodes)
        dest_node = st.selectbox(
            "Seleccionar Nodo Final", all_nodes, index=all_nodes.index(default_node)
        )
        submit_button = st.form_submit_button("Actualizar Grafo")

    # Crear una capa adicional para resaltar los nodos específicos
    origin_node_color = [255, 0, 0, 200]  # Rojo con transparencia
    dest_node_color = [0, 255, 0, 200]  # Verde con transparencia

    G.add_node_layer(orig_node, origin_node_color)
    G.add_node_layer(dest_node, dest_node_color)

    # Mostrar el grafo en Streamlit
    G.show()
    # Seleccionar Estrategias
    selected_strategy = st.sidebar.selectbox(
        "Selecciona Estrategia",
        ["e-greedy", "UCB1", "exp3 β constante", "exp3 β dinámico"],
    )

    if not selected_strategy:
        st.write("No se ha seleccionado estrategia.")
    else:
        st.write(f"Estrategia seleccionada: {selected_strategy}")

    get_qstar_button = st.button("Calcular Políticas Óptimas y Tabla Q*")
    # Crear un estado para policies y q_star
    if "policies" not in st.session_state:
        st.session_state.policies = None
    if "q_star" not in st.session_state:
        st.session_state.q_star = None

    if get_qstar_button:
        with st.spinner("Calculando políticas óptimas y tabla Q*..."):
            # Obtener la política óptima para cada nodo en el grafo hasta el destino
            policies = get_optimal_policy(G.G, dest_node)
            optimal_policy = policies
            print(f"\nPolítica óptima: {optimal_policy}\n")
            
            # Obtener la tabla Q* a partir de las políticas óptimas
            q_star = get_q_table_for_policy(G.G, policies, dest_node)
            # Guardar los resultados
            st.session_state.policies = policies
            st.session_state.q_star = q_star
        st.success("Políticas óptimas y tabla Q* calculadas!")

    # Usar st.form para agrupar sliders y botón de entrenamiento
    with st.form("training_form"):
        num_episodes = st.number_input(
            "Número de episodios",
            min_value=1000,
            max_value=1000000,
            value=30000,
            step=5000,
        )
        alpha = st.slider("Alpha", min_value=0.01, max_value=0.1, value=0.1, step=0.01)
        gamma = st.slider("Gamma", min_value=0.01, max_value=1.0, value=1.0, step=0.01)

        # Inicialización de variables para controlar la existencia de botón de envío
        submit_button = False
        epsilon = None
        c = None

        if selected_strategy == "e-greedy":
            epsilon = st.slider(
                "Epsilon", min_value=0.01, max_value=1.0, value=0.1, step=0.01
            )
            action_selector = EpsilonGreedyActionSelector(epsilon=epsilon)
        elif selected_strategy == "UCB1":
            c = st.slider(
                "C", min_value=1.0, max_value=5.0, value=2.0, step=0.5
            )  # Asegúrate de que todos los valores sean del mismo tipo (float).
            action_selector = UCB1ActionSelector(c=c)
        elif selected_strategy == "exp3 β dinámico":
            select_dynamic_beta = st.selectbox(
                "Selecciona el parámetro β dinámico",
                ["t", "t / T", "sqrt(t)", "log(t)"],
            )
            action_selector = Exp3ActionSelector(beta=select_dynamic_beta)

        elif selected_strategy == "exp3 β constante":
            fixed_beta = st.slider(
                "β", min_value=0.01, max_value=1.0, value=0.1, step=0.01
            )
            action_selector = Exp3ActionSelector(beta=fixed_beta)

        submit_button = st.form_submit_button("Entrenar Agente")

        if submit_button:
            with st.spinner("Entrenando el agente..."):
                strategy = "exp3" if "exp3" in selected_strategy else selected_strategy
                # Crear entorno
                env = SSPEnv(grafo=G.G, start_state=orig_node, terminal_state=dest_node)
                # Crear agente
                agent = QAgentSSP(
                    env, alpha=alpha, gamma=gamma, action_selector=action_selector
                )
                print(st.session_state.policies[orig_node])
                
                agent.train(
                    num_episodes,
                    q_star=st.session_state.q_star,
                    policy=st.session_state.policies[orig_node],
                    distribution="lognormal",
                )
            st.success("Entrenamiento completado!")
            
            with st.spinner("Guardando resultados..."):
                # Crear carpetas para guardar resultados
                strategies_list = ["e-greedy", "UCB1", "exp3"]
                for element in strategies_list:
                    temp_path = f"results/{location_name}/{orig_node}-{dest_node}/constant_alpha/{element}/"
                    # Si no existe la carpeta, crearla
                    if not os.path.exists(temp_path):
                        os.makedirs(temp_path)
                # Ruta para guardar resultados
                agent_storage_path = os.path.join(BASE_DIR, 'results/',f"{location_name}/{orig_node}-{dest_node}/constant_alpha/{strategy}/")
               
                # Si no existe la carpeta, crearla
                if not os.path.exists(agent_storage_path):
                    os.makedirs(agent_storage_path)
                # Guardar resultados
                save_model_results(
                    agent, nombre=f"QAgentSSP_{alpha:.2f}", path=agent_storage_path
                )
                

