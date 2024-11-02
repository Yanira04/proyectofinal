import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import requests
import os

# Tamaño de entrada de las imágenes
IMG_SIZE = (224, 224)

url = "https://drive.google.com/uc?export=download&id=1-ga1C9PZrBguuai001zoorhhcaUNFEg0"
response = requests.get(url, stream=True)

# Guardar el modelo en un archivo local
with open('modelo.h5', 'wb') as f:
    for chunk in response.iter_content(chunk_size=1024):
        if chunk:
            f.write(chunk)

try:
    model = tf.keras.models.load_model('modelo.h5')
    st.success("Pesos del modelo cargados correctamente.")
except Exception as e:
    st.error(f"Error al cargar los pesos del modelo: {e}")

# Verificación de carga de archivo
uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "jpeg", "png"], label_visibility="hidden")

if uploaded_file is not None:
    # Mostrar la imagen subida
    st.image(uploaded_file, caption="Imagen cargada", use_column_width=True)

    # Preprocesamiento de la imagen para hacer la predicción
    img = image.load_img(uploaded_file, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Realizar la predicción
    prediction = model.predict(img_array)

    # Determinar la clase basada en la predicción
    class_label = "NORMAL" if prediction[0][0] < 0.5 else "PNEUMONIA"
    confidence = prediction[0][0] if class_label == "PNEUMONIA" else 1 - prediction[0][0]

    # Mostrar resultados
    st.write(f"El modelo predice que la imagen es de un {class_label}.")
    st.write(f"Confianza de la predicción: {confidence:.4f}")