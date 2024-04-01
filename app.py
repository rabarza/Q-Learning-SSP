import streamlit as st
from city.app_results import ResultsVisualizer as ResultsVisualizerCity
import city.app_train_agents as app_train_agents
from perceptron.app import PerceptronApp
from perceptron.app import ResultsVisualizer as ResultsVisualizerPerceptron


st.sidebar.title("Navegación")
pagina = st.sidebar.radio("Selecciona una página:", ["Ciudad", "Perceptron"])
if pagina == "Ciudad":
    selected_option = st.radio(
        "Selecciona una página:", ["Entrenar Agente", "Ver Resultados"]
    )
    if selected_option == "Ver Resultados":
        results_visualizer = ResultsVisualizerCity()
        results_visualizer.load_results()
        results_visualizer.show_results()
    else:
        app_train_agents.show()
else:
    selected_option = st.radio(
        "Selecciona una página:", ["Crear Red", "Entrenar Agente", "Ver Resultados"]
    )
    app = PerceptronApp()
    if selected_option == "Crear Red":
        app.create_graph_helper()
    elif selected_option == "Entrenar Agente":
        app.train_agents_view()
    else:
        results_visualizer = ResultsVisualizerPerceptron()
        results_visualizer.load_results()
        results_visualizer.show_results()
