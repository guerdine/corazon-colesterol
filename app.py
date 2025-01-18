import streamlit as st
import joblib as jb
import numpy as np

# Cargar el modelo y el escalador
model = jb.load("nb_model.bin")
scaler = jb.load("scaler.pkl")

# Título de la aplicación
st.title("Aplicación para predecir si padece del corazón")
st.subheader("Prediccion de riesgo cardiovascular")
st.markdown("**Autor:** Daniel Vasquez")

# Barra lateral con controles
st.sidebar.header("Parámetros de entrada")

# Slider para la edad
edad = st.sidebar.slider(
    "Edad (años)", 
    min_value=20, 
    max_value=80, 
    value=40, 
    step=1
)

# Entrada numérica para el colesterol
colesterol = st.sidebar.slider(
    "Colesterol (mg/dL)", 
    min_value=100, 
    max_value=600, 
    value=200, 
    step=10
)

# Botón para hacer predicciones
if st.sidebar.button("Predecir"):
    # Crear un array con las características de entrada
    features = np.array([[edad, colesterol]])

    # Escalar las características de entrada
    scaled_features = scaler.transform(features)

    # Hacer la predicción
    prediction = model.predict(scaled_features)[0]

    # Mostrar el resultado
    if prediction == 0:
        st.success("No sufrirá del corazón. ¡Sigue cuidando tu salud!")
    else:
        st.warning("Advertencia: Puede sufrir del corazón. Por favor, consulte con un médico.")

# Información adicional en la página principal
st.write(
    "Esta herramienta utiliza un modelo de Machine Learning entrenado para predecir si una persona podría sufrir problemas cardíacos en función de su edad y nivel de colesterol."
)