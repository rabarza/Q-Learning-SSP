import streamlit as st

def initialize_session_state_variables(*args, **kwargs):
    '''Inicializa las variables de estado de la sesión si no existen.
    
    Esta función se utiliza para inicializar las variables de estado de la sesión
    si no existen. Las variables de estado de la sesión se utilizan para almacenar
    los resultados de los cálculos realizados en la aplicación. Estas variables
    persisten en la sesión de Streamlit y se pueden utilizar para mostrar los
    resultados en la aplicación.
    '''    
    for arg in args:
        print(arg)
        if arg not in st.session_state:
            st.session_state[arg] = None
    

def clear_session_state_variables():
    st.session_state.graph = None
    st.session_state.policies = None
    st.session_state.q_star = None
    st.session_state.q_star_serialized = None