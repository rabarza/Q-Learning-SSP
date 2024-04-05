import sys

sys.path.append("../RLib")  # Agrega el directorio RLib al sys.path
sys.path.append("../city")  # Agrega el directorio city al sys.path

import os
import time
import streamlit as st
from city.app_graph import CityGraphPlotter, QUERIE_PARAMS

# from .app_results import city_selectbox
from time import sleep

from RLib.environments.ssp import SSPEnv
from RLib.agents.ssp import QAgentSSP
from RLib.utils.dijkstra_utils import (
    get_optimal_policy,
    get_shortest_path_from_policy,
    get_q_table_for_policy,
)
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

    # Solo actualizar el grafo si la ciudad seleccionada cambia
    if st.session_state.get("location_name") != location_name:
        G = CityGraphPlotter(QUERIE_PARAMS, query=location_name)
        st.session_state.location_name = location_name
        st.session_state.graph = G
    else:
        G = st.session_state.graph

    # Visualizar en Streamlit
    st.markdown(
        f"<h1>Entorno Seleccionado: {location_name}. ({len(G.graph.nodes())} Nodos)</h1>",
        unsafe_allow_html=True,
    )

    # Interfaz para seleccionar el nodo inicial y final
    all_nodes = list(G.graph.nodes)
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
    st.session_state.graph.show()

    # Crear un estado para policies y optimal_q_table
    if "shortest_path" not in st.session_state:
        st.session_state.shortest_path = None
    if "optimal_q_table" not in st.session_state:
        st.session_state.optimal_q_table = None
    if "optimal_policy" not in st.session_state:
        st.session_state.optimal_policy = None
    if "submit_button_strategy" not in st.session_state:
        st.session_state.submit_button_strategy = False
    if "submit_button_train_agent" not in st.session_state:
        st.session_state.submit_button_train_agent = False
        
    get_qstar_button = st.button("Calcular Políticas Óptimas y Tabla Q*")
    if get_qstar_button:
        with st.spinner("Calculando políticas óptimas y tabla Q*..."):
            # Obtener la política óptima para cada nodo en el grafo hasta el destino
            optimal_policy = get_optimal_policy(G.graph, dest_node)
            shortest_path = get_shortest_path_from_policy(
                optimal_policy, orig_node, dest_node
            )
            print(f"\nPolítica óptima: {optimal_policy}\n")

            # Obtener la tabla Q* a partir de las políticas óptimas
            optimal_q_table = get_q_table_for_policy(G.graph, optimal_policy, dest_node)
            # Guardar los resultados
            st.session_state.optimal_policy = optimal_policy
            st.session_state.optimal_q_table = optimal_q_table
            st.session_state.shortest_path = shortest_path
        st.success("Políticas óptimas y tabla Q* calculadas!")

    # Interfaz para seleccionar la estrategia de selección de acción
    with st.form("strategy_form"):
        st.write("Seleccionar Tipo de aprendizaje:")

        selected_strategy = st.selectbox(
            "Selecciona Estrategia",
            ["e-greedy", "UCB1", "exp3 β constante", "exp3 β dinámico"],
        )

        alpha_type = st.selectbox(
            "Selecciona el tipo de tasa de aprendizaje α",
            ["constante", "dinámico"],
        )

        st.session_state.alpha_type = alpha_type
        st.session_state.selected_strategy = selected_strategy

        submit_button_strategy = st.form_submit_button("Actualizar Estrategia")
        if submit_button_strategy:
            st.session_state.submit_button_strategy = True

    # Usar st.form para agrupar sliders y botón de entrenamiento
    if st.session_state.submit_button_strategy:
        with st.form("training_form"):
            num_episodes = st.number_input(
                "Número de episodios",
                min_value=1000,
                max_value=1000000,
                value=30000,
                step=5000,
            )

            if st.session_state.alpha_type == "constante":
                alpha = st.slider(
                    "Learning rate α", min_value=0.01, max_value=0.1, value=0.1, step=0.01
                )
            else:
                alpha = st.selectbox(
                    "Learning rate α",
                    ["1/N(s,a)", "max(0.01, 1/N(s,a))", "1/sqrt(N(s,a))", "1/log(N(s,a))"],
                )
            gamma = st.slider(
                "Discount Rate γ", min_value=0.01, max_value=1.0, value=1.0, step=0.01
            )

            epsilon = None
            c = None

            if selected_strategy == "e-greedy":
                epsilon = st.slider(
                    "Epsilon", min_value=0.01, max_value=1.0, value=0.1, step=0.01
                )
                action_selector = EpsilonGreedyActionSelector(epsilon=epsilon)
            elif selected_strategy == "UCB1":
                c = st.slider(
                    "C", min_value=1.0, max_value=15.0, value=2.0, step=0.5
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

            submit_button_train_agent = st.form_submit_button("Entrenar Agente")
            st.session_state.submit_button_train_agent = submit_button_train_agent

        if st.session_state.submit_button_train_agent:
            with st.spinner(f"Entrenando el agente... {selected_strategy}"):
                strategy = "exp3" if "exp3" in selected_strategy else selected_strategy
                # Crear entorno
                env = SSPEnv(
                    grafo=G.graph, start_state=orig_node, terminal_state=dest_node
                )
                # Crear agente
                agent = QAgentSSP(
                    env, alpha=alpha, gamma=gamma, action_selector=action_selector
                )

                agent.train(
                    num_episodes,
                    shortest_path=st.session_state.shortest_path,
                    distribution="lognormal",
                    q_star=st.session_state.optimal_q_table,
                )
                # Asignar la política óptima al agente
                agent.optimal_policy = st.session_state.optimal_policy

            st.success("Entrenamiento completado!")

            with st.spinner("Guardando resultados..."):
                # Crear carpetas para guardar resultados
                strategies_list = ["e-greedy", "UCB1", "exp3"]
                for element in strategies_list:
                    temp_path = f"results/{location_name}/{orig_node}-{dest_node}/constant_alpha/{element}/"
                    files_dir = os.path.join(BASE_DIR, temp_path)
                    # Si no existe la carpeta, crearla
                    if not os.path.exists(files_dir):
                        os.makedirs(files_dir)
                # Ruta para guardar resultados
                agent_storage_path = os.path.join(
                    BASE_DIR,
                    "results/",
                    f"{location_name}/{orig_node}-{dest_node}/constant_alpha/{strategy}/",
                )

                # Si no existe la carpeta, crearla
                if not os.path.exists(agent_storage_path):
                    os.makedirs(agent_storage_path)
                # Guardar resultados
                save_model_results(
                    agent, nombre=f"QAgentSSP_{alpha:.2f}", path=agent_storage_path
                )
                
                st.session_state.submit_button_strategy = False
            st.success("Resultados guardados!")
            time.sleep(2)
            st.experimental_rerun()
