import sys

sys.path.append("../RLib")  # Agrega el directorio RLib al sys.path
sys.path.append("../city")  # Agrega el directorio city al sys.path

import streamlit as st
import os
import json
from city.app_graph import CityGraphPlotter, QUERIE_PARAMS
from RLib.utils.files import load_model_results, find_files_by_keyword
from RLib.utils.plots import plot_results_per_episode_comp_plotly
from RLib.utils.serializers import QAgentSSPSerializer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")


def get_selected_agents(
    selected_alpha_type, selected_strategies, agents_const, agents_dyna
):
    is_constant = "Constante" in selected_alpha_type
    is_dynamic = "Dinámica" in selected_alpha_type

    if is_constant:
        with st.form(key="alpha_form"):
            min_alpha, max_alpha = st.slider(
                "Selecciona un rango de valores alpha",
                min_value=0.01,
                max_value=0.1,
                value=(0.08, 0.1),
                step=0.01,
            )
            apply_button = st.form_submit_button(label="Aplicar")

        st.write(f"Valor mínimo de alpha: {min_alpha}")
        st.write(f"Valor máximo de alpha: {max_alpha}")

        selected_agents_fixed_alpha = [
            agent
            for agent in agents_const
            if agent.strategy in selected_strategies
            and min_alpha <= agent.alpha <= max_alpha
        ]
    else:
        selected_agents_fixed_alpha = []

    if is_dynamic:
        selected_agents_dynamic_alpha = [
            agent for agent in agents_dyna if agent.strategy in selected_strategies
        ]
    else:
        selected_agents_dynamic_alpha = []

    if is_constant and is_dynamic:
        selected_agents = selected_agents_fixed_alpha + selected_agents_dynamic_alpha
    elif is_constant:
        selected_agents = selected_agents_fixed_alpha
    elif is_dynamic:
        selected_agents = selected_agents_dynamic_alpha
    else:
        selected_agents = []

    return selected_agents


class ResultsVisualizer:
    def __init__(self):
        self.location_name = ""
        self.ruta = ""
        self.agents_const = []
        self.agents_dyna = []
        self.agents = []

    @staticmethod
    @st.cache_data(persist=True)
    def load_css(file_name, folder="styles"):
        with open(f"{folder}/{file_name}") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    @staticmethod
    @st.cache_data(persist=True)
    def load_agents(ruta_carpeta):
        agent_keywords = {"e-greedy": "e-greedy", "UCB1": "UCB1", "exp3": "exp3"}
        greedy_files = find_files_by_keyword("e-", ruta_carpeta + "e-greedy")
        print(greedy_files)
        greedy_models = list(
            map(
                lambda x: load_model_results(x, ruta_carpeta + "e-greedy"), greedy_files
            )
        )
        ucb_files = find_files_by_keyword("UCB1", ruta_carpeta + "UCB1")
        ucb_models = list(
            map(lambda x: load_model_results(x, ruta_carpeta + "UCB1"), ucb_files)
        )
        exp3_files = find_files_by_keyword("exp3", ruta_carpeta + "exp3")
        exp3_models = list(
            map(lambda x: load_model_results(x, ruta_carpeta + "exp3"), exp3_files)
        )
        return greedy_models + ucb_models + exp3_models

    @staticmethod
    def city_selectbox(QUERIE_PARAMS, key_suffix=""):
        location_name = st.sidebar.selectbox(
            "Seleccionar Ciudad",
            list(QUERIE_PARAMS.keys()),
            key=f"city_select_{key_suffix}",
        )
        ruta_carpeta = f"results/{location_name}/"
        return location_name, ruta_carpeta

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
        print("selected_agent", selected_agent)
        # Serializar el agente seleccionado
        serialized_agent = QAgentSSPSerializer(selected_agent).to_dict()
        st.write(serialized_agent)

    def load_results(self):

        self.location_name, self.ruta = self.city_selectbox(
            QUERIE_PARAMS, key_suffix="main_results"
        )
        self.ruta = os.path.join(BASE_DIR, self.ruta)
        if not os.path.exists(self.ruta):
            st.error(f"No se encontraron resultados para {self.ruta}")
            return

        sub_folders = [
            carpeta
            for carpeta in os.listdir(self.ruta)
            if os.path.isdir(os.path.join(self.ruta, carpeta))
        ]
        subcarpeta = st.sidebar.selectbox("Seleccionar entrenamiento", sub_folders)

        agents_path = os.path.join(BASE_DIR, self.ruta + subcarpeta)
        ruta_carpeta_const = agents_path + "/constant_alpha/"
        ruta_carpeta_dyna = agents_path + "/dynamic_alpha/"

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

    def show_results(self):
        self.load_css("styles.css")

        G = CityGraphPlotter(QUERIE_PARAMS, query=self.location_name)
        if not self.agents:
            st.title(f"No hay resultados para la ciudad de {self.location_name}")
            G.show()
        else:
            orig_node = self.agents[0].env.start_state
            dest_node = self.agents[0].env.terminal_state
            origin_node_color = [255, 0, 0, 200]
            dest_node_color = [0, 255, 0, 200]
            G.add_node_layer(orig_node, origin_node_color)
            G.add_node_layer(dest_node, dest_node_color)

            st.title(
                "Resultados de Q-Learning para el Stochastic Shortest Path Problem (SSP)"
            )
            st.markdown(
                f"<h1>Mapa Seleccionado: {self.location_name}</h1>",
                unsafe_allow_html=True,
            )

            legend_html = """
                <div style="margin: 10px; padding: 10px; border: 1px solid #ccc; display: flex; justify-content: space-around; ">
                    <div style="display: flex; ">
                        <div style="margin-right: 10px;">Nodo inicial:</div> 
                        <div class="red-circle"></div>
                    </div>
                    <div style="display: flex; ">
                        <div style="margin-right: 10px;">Nodo final:</div> 
                        <div class="green-circle"></div>
                    </div>
                </div>
            """

            st.markdown(legend_html, unsafe_allow_html=True)
            G.show()

            default_options = {
                "strategies": ["e-greedy", "UCB1", "exp3"],
                "alpha_type": ["Constante", "Dinámica"],
            }

            selected_strategies = st.sidebar.multiselect(
                "Selecciona Estrategias",
                ["e-greedy", "UCB1", "exp3"],
                default=default_options["strategies"],
            )
            selected_strategies.sort()

            selected_alpha_type = st.sidebar.multiselect(
                "Selecciona Tipo de learning rate",
                ["Constante", "Dinámica"],
                default=default_options["alpha_type"],
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
                print(selected_agents)

                for criteria in [
                    "error",
                    "policy error",
                    "steps",
                    "score",
                    "regret",
                    # "cumulative regret",
                ]:
                    fig = plot_results_per_episode_comp_plotly(
                        selected_agents, criteria
                    )
                    st.write(fig)
                # Mostrar los resultados serializados
                # self.show_serialized_agents(selected_agents)
                # Descargar los resultados serializados en un archivo JSON
                st.download_button(
                    label="Descargar resultados serializados",
                    data=json.dumps(
                        [
                            QAgentSSPSerializer(agent).to_dict()
                            for agent in selected_agents
                        ]
                    ),
                    file_name="results.json",
                    mime="application/json",
                )


if __name__ == "__main__":
    results_visualizer = ResultsVisualizer()
    results_visualizer.load_results()
    results_visualizer.show_results()
