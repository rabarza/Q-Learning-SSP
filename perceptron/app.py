import os
import time
import random
import json
import streamlit as st
import networkx as nx

from perceptron.utils.state import (
    initialize_session_state_variables,
    clear_session_state_variables,
)
from perceptron.utils.plots import plot_perceptron_graph
from perceptron.utils.files import (
    load_graph,
    save_graph,
    get_pkl_files_in_folder,
    get_folders_list,
)
from perceptron.utils.serializers import serialize_dict_of_dicts
from train.actions import get_action_selector

# TRAIN library
from RLib.environments.ssp import SSPEnv
from RLib.agents.ssp import QAgentSSP
from RLib.utils.dijkstra_utils import (
    get_optimal_policy,
    get_shortest_path_from_policy,
    get_q_table_for_policy,
)

from RLib.utils.file_utils import save_model_results
from RLib.utils.file_utils import load_model_results, find_files_by_keyword
from RLib.utils.plot_utils import plot_results_per_episode_comp_plotly

# Importar serializadores
from RLib.utils.serializers import QAgentSSPSerializer


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GRAPHS_DIR_NAME = "networks"


def get_selected_agents(
    selected_alpha_type, selected_strategies, agents_const, agents_dyna
):
    selected_agents_fixed_alpha = []
    selected_agents_dynamic_alpha = []

    if "Constante" in selected_alpha_type:
        with st.form(key="alpha_form"):
            min_alpha, max_alpha = st.slider(
                "Selecciona un rango de valores alpha",
                min_value=0.01,
                max_value=0.1,
                value=(0.08, 0.1),
                step=0.01,
            )
            apply_button = st.form_submit_button("Aplicar")

        st.write(f"Valor mínimo de alpha: {min_alpha}")
        st.write(f"Valor máximo de alpha: {max_alpha}")

        selected_agents_fixed_alpha = [
            agent
            for agent in agents_const
            if agent.strategy in selected_strategies
            and min_alpha <= agent.alpha <= max_alpha
        ]

    if "Dinámica" in selected_alpha_type:
        selected_agents_dynamic_alpha = [
            agent for agent in agents_dyna if agent.strategy in selected_strategies
        ]
        st.latex(rf""" \alpha_t = {selected_agents_dynamic_alpha[0].alpha_formula} """)

    if "Constante" in selected_alpha_type and "Dinámica" in selected_alpha_type:
        selected_agents = selected_agents_fixed_alpha + selected_agents_dynamic_alpha
    elif "Constante" in selected_alpha_type:
        selected_agents = selected_agents_fixed_alpha
    elif "Dinámica" in selected_alpha_type:
        selected_agents = selected_agents_dynamic_alpha
    else:
        selected_agents = []

    return selected_agents


class ResultsVisualizer:
    def __init__(self):
        self.agents = []

    @staticmethod
    def load_agents(ruta_carpeta):
        agent_keywords = {"e-greedy": "e-greedy", "UCB1": "UCB1", "exp3": "exp3"}
        greedy_files = find_files_by_keyword("e-", ruta_carpeta + "e-greedy")
        print(f"\nArchivos greedy encontrados: {greedy_files}\n")
        greedy_models = list(
            map(
                lambda x: load_model_results(x, ruta_carpeta + "e-greedy"), greedy_files
            )
        )
        print(f"\nModelos greedy encontrados: {greedy_models}\n")
        ucb_files = find_files_by_keyword("UCB1", ruta_carpeta + "UCB1")
        ucb_models = list(
            map(lambda x: load_model_results(x, ruta_carpeta + "UCB1"), ucb_files)
        )
        exp3_files = find_files_by_keyword("exp3", ruta_carpeta + "exp3")
        exp3_models = list(
            map(lambda x: load_model_results(x, ruta_carpeta + "exp3"), exp3_files)
        )
        return greedy_models + ucb_models + exp3_models

    def load_results(self):
        # Ruta donde se encuentran los resultados
        file_name = st.selectbox(
            "Selecciona un grafo existente", get_folders_list(BASE_DIR)
        )
        # Agregar la extensión .pkl para poder cargar el archivo
        file_name += ".pkl"

        if st.session_state.get("grafo_perceptron"):
            del st.session_state["grafo_perceptron"]
        with st.spinner("Cargando..."):
            G = load_graph(file_name, base_dir=BASE_DIR, folder_name=GRAPHS_DIR_NAME)
            fig = plot_perceptron_graph(G)
            st.write(fig)
            st.success("Cargado!", icon="✅")

        # Ruta donde se encuentran los resultados
        ruta_resultados = os.path.join(BASE_DIR, "results", file_name.split(".")[0])
        agents_path = ruta_resultados
        print(f"\nRuta de los agentes: {agents_path}\n")
        ruta_carpeta_const = os.path.join(agents_path, "constant_alpha/")
        ruta_carpeta_dyna = os.path.join(agents_path, "dynamic_alpha/")

        # Cargar los agentes (de learning rate constante y dinámica) si existen
        self.agents_const = (
            self.load_agents(ruta_carpeta_const)
            if os.path.exists(ruta_carpeta_const)
            else []
        )
        self.agents_dyna = (
            self.load_agents(ruta_carpeta_dyna)
            if os.path.exists(ruta_carpeta_dyna)
            else []
        )
        self.agents = self.agents_const + self.agents_dyna

    def show_serialized_agents(self, selected_agents):
        # Crear un diccionario para asociar el ID único con la representación de cadena
        agent_options = {str(agent.id): str(agent) for agent in selected_agents}
        # Selección de agente
        selected_agent_id = st.selectbox(
            "Selecciona un agente",
            list(agent_options.keys()),  # Usar el ID único como valor de la opción
            format_func=lambda x: agent_options[
                x
            ],  # Mostrar la representación de cadena en la interfaz
            key="selected_agent",
        )
        # get the selected agent as agent object
        selected_agent = next(
            agent for agent in selected_agents if str(agent.id) == selected_agent_id
        )

        # Serializar el agente seleccionado
        serialized_agent = QAgentSSPSerializer(selected_agent).to_dict()
        st.write(serialized_agent)

    def show_results(self):
        default_options = {
            "strategies": ["e-greedy", "UCB1", "exp3"],
            "alpha_type": ["Constante"],
        }

        selected_strategies = st.sidebar.multiselect(
            "Selecciona Estrategias",
            ["e-greedy", "UCB1", "exp3"],
            default=default_options["strategies"],
            key="strategies",
        )
        selected_strategies.sort()

        selected_alpha_type = st.sidebar.multiselect(
            "Selecciona Tipo de Alpha",
            ["Constante", "Dinámica"],
            default=default_options["alpha_type"],
            key="alpha_type",
        )

        if not selected_strategies:
            st.write("No se han seleccionado estrategias.")
        else:
            st.write(f"Estrategias seleccionadas: {selected_strategies}")
            selected_agents = get_selected_agents(
                selected_alpha_type,
                selected_strategies,
                self.agents_const,
                self.agents_dyna,
            )
            if not selected_agents:
                return

            for criteria in ["error", "policy error", "steps", "score"]:
                fig = plot_results_per_episode_comp_plotly(selected_agents, criteria)
                st.write(fig)
            # Mostrar los resultados del agente seleccionado
            # self.show_serialized_agents(selected_agents)
            st.download_button(
                label="Descargar resultados serializados",
                data=json.dumps(
                    [QAgentSSPSerializer(agent).to_dict() for agent in selected_agents]
                ),
                file_name="results.json",
                mime="application/json",
            )


class PerceptronApp:
    def __init__(self):
        self.capas = ["Entrada", "Salida"]
        self.nodos_por_capa = [1, 1]
        self.numero_capas_ocultas = 1
        self.posiciones = {}
        self.fig = None

    def __str__(self):
        return f"PerceptronApp {list(zip(self.capas, self.nodos_por_capa))}, {self.numero_capas_ocultas}"

    def crear_grafo_perceptron(self):
        """Crea un grafo de NetworkX con las capas y nodos del perceptrón.
        Las conexiones entre nodos son aleatorias y se asigna una longitud a cada conexión en forma aleatoria.
        La longitud se genera con un número entero aleatorio entre 1 y 100.

        Returns:
        --------------------------------
            grafo_perceptron (nx.DiGraph): Grafo de NetworkX con las capas y nodos del perceptrón.
        """
        grafo_perceptron = nx.DiGraph()

        # Recorrer capas: (0, 'Entrada') (1, 'Oculta 1') ... (0, 'Salida')
        for i, capa in enumerate(self.capas):
            for j in range(self.nodos_por_capa[i]):
                # Asignar posición a cada nodo (importante para visualización)
                self.posiciones[(capa, j)] = (
                    i * 1.5,
                    j * 1 - (self.nodos_por_capa[i] - 1) / 2,
                )
                # Añadir nodo al grafo con la posición correspondiente y un largo aleatorio
                grafo_perceptron.add_node(
                    (capa, j),
                    pos=self.posiciones[(capa, j)],
                )

        for i in range(len(self.capas) - 1):
            for j in range(self.nodos_por_capa[i]):
                for k in range(self.nodos_por_capa[i + 1]):
                    grafo_perceptron.add_edge(
                        (self.capas[i], j),
                        (self.capas[i + 1], k),
                        length=random.randint(100, 200),
                    )

        # Añadir conexion redundante en el último nodo (nodo terminal)
        grafo_perceptron.add_edge(
            (self.capas[-1], self.nodos_por_capa[-1] - 1),
            (self.capas[-1], self.nodos_por_capa[-1] - 1),
            length=0,
        )
        return grafo_perceptron

    def get_grafo_perceptron(self):
        return self.crear_grafo_perceptron()

    def refresh_graph_display(self):
        """Actualiza la interfaz de usuario con los nodos y conexiones del grafo."""

        for i in range(self.numero_capas_ocultas):
            capa_actual = i + 1  # Comienza en 1
            self.capas.insert(-1, f"Oculta {capa_actual}")
            key_suffix = f"{capa_actual}_{i}"
            self.nodos_por_capa.insert(
                -1,
                st.number_input(
                    f"Nodos capa {capa_actual}",
                    min_value=1,
                    key=f"Nodos_x_capa_{key_suffix}",
                ),
            )

        grafo_perceptron = self.crear_grafo_perceptron()
        fig = plot_perceptron_graph(grafo_perceptron)
        # fig = plot_perceptron_graph(grafo_perceptron)
        st.write(fig)

    def create_graph_helper(self):
        """
        Helper method to create a graph for the perceptron app.

        This method displays a sidebar in the app interface where the user can add hidden layers to the graph.
        It also provides an option to save the graph.

        Returns:
            None
        """
        with st.sidebar:
            st.header("Agregar capas ocultas")
            self.numero_capas_ocultas = st.number_input(
                "Número de capas ocultas",
                min_value=0,
                value=2,
                key=f"numero_capas_ocultas{self.numero_capas_ocultas}",
            )

        self.refresh_graph_display()

        if st.button("Guardar grafo"):
            grafo = self.get_grafo_perceptron()
            if save_graph(self, grafo, BASE_DIR):
                st.success(f"Grafo guardado")
            else:
                st.error("Error al guardar el grafo")

    def load_graph_helper(self):
        """Cargar un grafo guardado en la carpeta 'networks' y mostrarlo en la interfaz de usuario.
        Almacena el grafo en el estado de la sesión, mediante la variable 'graph', y almacena el nombre del archivo en 'graph_name'.
        """
        variables = ["graph", "file_name"]
        initialize_session_state_variables(variables)
        # Obtener la lista de archivos .pkl en la carpeta 'networks'
        try:
            files = get_pkl_files_in_folder(BASE_DIR, GRAPHS_DIR_NAME)
            # Obtener la lista de nombres de archivos sin la extensión .pkl
            file_names = list(map(lambda x: x.split(".")[0], files))
            # Seleccionar un grafo guardado
            file_name = st.selectbox(
                "Selecciona un grafo existente",
                file_names,
                on_change=clear_session_state_variables,
            )
            file_name += ".pkl"  # Agregar la extensión .pkl

            with st.spinner("Cargando..."):
                G = load_graph(
                    file_name, base_dir=BASE_DIR, folder_name=GRAPHS_DIR_NAME
                )

                fig = plot_perceptron_graph(G)
                st.write(fig)
                st.success("Cargado!", icon="✅")
                # Almacenar el grafo en el estado de la sesión
                st.session_state.graph_name = file_name
                st.session_state.graph = G
        except Exception as e:
            st.error(
                f"No hay grafos guardados en la carpeta {GRAPHS_DIR_NAME}. Crea un grafo nuevo para poder entrenar agentes."
            )

    def train_agents_view(self):
        variables = [
            "graph",
            "policies",
            "optimal_q_table",
            "q_star_serialized",
            "optimal_policy",
            "shortest_path",
        ]
        initialize_session_state_variables(variables)

        # Seleccionar un grafo guardado
        self.load_graph_helper()
        G = st.session_state.graph
        file_name = st.session_state.graph_name

        # Definir nodos de origen y destino
        orig_node = ("Entrada", 0)
        dest_node = ("Salida", 0)

        st.write("Nodo de origen:", orig_node)
        st.write("Nodo de destino:", dest_node)

        # Crear un estado para botones de formularios
        if "submit_button_strategy" not in st.session_state:
            st.session_state.submit_button_strategy = False
        if "submit_button_train_agent" not in st.session_state:
            st.session_state.submit_button_train_agent = False

        get_qstar_button = st.button("Calcular Políticas Óptimas y Tabla Q*")
        if get_qstar_button:
            with st.spinner("Calculando políticas óptimas y tabla Q*..."):
                # Obtener la política óptima para cada nodo en el grafo hasta el destino
                optimal_policy = get_optimal_policy(G, dest_node)
                shortest_path = get_shortest_path_from_policy(
                    optimal_policy, orig_node, dest_node
                )
                print(f"\nPolítica óptima: {optimal_policy}\n")

                # Obtener la tabla Q* completa a partir de las políticas óptimas
                optimal_q_table = get_q_table_for_policy(G, optimal_policy, dest_node)
                serialized_qtable = serialize_dict_of_dicts(optimal_q_table)

                # Guardar los resultados en el estado de la sesión
                # Importante para mantener persistencia en datos de la sesión
                st.session_state.optimal_policy = optimal_policy
                st.session_state.optimal_q_table = optimal_q_table
                st.session_state.q_star_serialized = serialized_qtable
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

        if st.session_state.submit_button_strategy:
            with st.form("training_form"):
                num_episodes = st.number_input(
                    "Número de episodios",
                    min_value=1000,
                    max_value=1000000,
                    value=30000,
                    step=5000,
                )
                # Selección de tasa de aprendizaje α
                if st.session_state.alpha_type == "constante":
                    alpha = st.slider(
                        "Learning rate α",
                        min_value=0.01,
                        max_value=0.1,
                        value=0.1,
                        step=0.01,
                    )
                else:
                    alpha = st.selectbox(
                        "Learning rate α",
                        [
                            "1/N(s,a)",
                            "max(0.01, 1/N(s,a))",
                            "1/sqrt(N(s,a))",
                            "1/log(N(s,a))",
                        ],
                    )
                # Selección de factor de descuento γ
                gamma = st.slider(
                    "Discount Rate γ",
                    min_value=0.01,
                    max_value=1.0,
                    value=1.0,
                    step=0.01,
                )

                action_selector = get_action_selector(selected_strategy)
                submit_button_train_agent = st.form_submit_button("Entrenar Agente")
                st.session_state.submit_button_train_agent = submit_button_train_agent

        # Comenzar entrenamiento
        if st.session_state.submit_button_train_agent:
            with st.spinner(f"Entrenando el agente... {selected_strategy}"):
                strategy = "exp3" if "exp3" in selected_strategy else selected_strategy
                # Crear entorno
                env = SSPEnv(grafo=G, start_state=orig_node, terminal_state=dest_node)
                # Crear agente
                agent = QAgentSSP(
                    env, alpha=alpha, gamma=gamma, action_selector=action_selector
                )
                # Entrenar agente
                agent.train(
                    num_episodes,
                    shortest_path=st.session_state.shortest_path,
                    q_star=st.session_state.optimal_q_table,
                    distribution="lognormal",
                )
                # Asignar la política óptima al agente
                agent.optimal_policy = st.session_state.optimal_policy
            st.success("Entrenamiento completado!")

            # Guardar resultados
            with st.spinner("Guardando resultados..."):
                # Crear carpetas para guardar resultados
                strategies_list = ["e-greedy", "UCB1", "exp3"]

                for element in strategies_list:
                    temp_path = (
                        f"results/{file_name.split('.')[0]}/constant_alpha/{element}/"
                    )
                    results_dir = os.path.join(BASE_DIR, temp_path)
                    # Si no existe la carpeta, crearla
                    if not os.path.exists(results_dir):
                        os.makedirs(results_dir)

                # Ruta para guardar resultados
                agent_storage_path = os.path.join(
                    BASE_DIR,
                    f"results/{file_name.split('.')[0]}/constant_alpha/{strategy}/",
                )
                # Si no existe la carpeta, crearla
                if not os.path.exists(agent_storage_path):
                    os.makedirs(agent_storage_path)

                # Guardar resultados
                save_model_results(
                    agent,
                    nombre=f"QAgentSSP_",
                    path=agent_storage_path,
                )
                st.session_state.submit_button_strategy = False
                st.session_state.submit_button_train_agent = False
            st.success("Resultados guardados!")
            time.sleep(2)
            st.experimental_rerun()


if __name__ == "__main__":
    app = PerceptronApp()
    app.show()
