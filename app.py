import json
from pathlib import Path
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# Configuración inicial de la interfaz
st.set_page_config(page_title="Clasificador de Flores IA", layout="centered")

# Encabezado personalizado con tus datos para la entrega de la tarea
st.title("🌻 Modelo predictivo de Flores - Clase de Inteligencia Artificial")
st.subheader("Estudiante: Edy Yandel Martinez Tejeda - Cuenta: 20221900004")
st.write("Suba una imagen para clasificar las flores con el modelo MobileNetV2 entrenado")

# Dimensiones estándar requeridas por MobileNetV2
IMG_SIZE = (224, 224)

# Configuración de la ruta relativa del modelo en la raíz de tu repositorio
MODEL_PATH = Path("flower_model.h5")

# Lista fija con el orden alfabético estricto que genera el ImageDataGenerator del entrenamiento
CLASES_ENTRENAMIENTO = ["daisy", "dandelion", "rose", "sunflower", "tulip"]

# Diccionario de traducción para las 5 categorías del dataset Flowers Recognition de Kaggle
LABELS_ES = {
    "daisy": "Margarita 🌼",
    "dandelion": "Diente de León 🌾",
    "rose": "Rosa 🌹",
    "sunflower": "Girasol 🌻",
    "tulip": "Tulipán 🌷"
}

@st.cache_resource
def cargar_modelo():
    if MODEL_PATH.exists():
        # Cargamos el archivo .h5 generado en el entrenamiento
        return tf.keras.models.load_model(MODEL_PATH, compile=False)
    st.error("No se encontró el modelo. Asegúrate de tener 'flower_model.h5' en la misma carpeta que app.py.")
    st.stop()
    
with st.sidebar:
    st.write("### Modo Depuración (Debug)")
    st.write("Si el modelo falla, tu entrenamiento guardó un orden de clases distinto.")

def preparar_imagen(img):
    # Asegurar formato RGB y tamaño 224x224
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    
    # Procesamiento idéntico al del nuevo entrenamiento
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    
    return np.expand_dims(arr, axis=0)

def predecir(img):
    # Tomar la salida directa de la capa Softmax del modelo entrenado
    probabilidades = modelo.predict(preparar_imagen(img), verbose=0)[0]
    top_indices = np.argsort(probabilidades)[::-1]
    
    return [
        (LABELS_ES[CLASES_ENTRENAMIENTO[i]], float(probabilidades[i]) * 100)
        for i in top_indices
    ]

# Cargar el modelo en memoria utilizando la caché de Streamlit
modelo = cargar_modelo()

# Componente interactivo para cargar archivos en la web
archivo = st.file_uploader("Seleccione una imagen de una flor", type=["jpg", "jpeg", "png"])

if archivo:
    # Cargar y desplegar la imagen en la interfaz gráfica
    imagen = Image.open(archivo)
    st.image(imagen, caption="Imagen analizada", use_container_width=True)

    with st.spinner("Inteligencia Artificial analizando..."):
        # Realizar el análisis con la red neuronal
        resultados = predecir(imagen)
    
    st.subheader("Resultado del Análisis")
    # Mostrar la predicción ganadora con un recuadro verde de éxito
    st.success(f"Predicción principal: **{resultados[0][0]}** ({resultados[0][1]:.2f}%)")

    # Mostrar la lista comparativa de probabilidades de la IA
    st.write("### Probabilidades del modelo:")
    for clase, prob in resultados:
        st.write(f"**{clase}**: {prob:.2f}%")
        # Barra de progreso visual integrada para mejorar el diseño de la entrega
        st.progress(prob / 100.0)
else:
    st.info("Cargue una imagen de una flor (Margarita, Diente de león, Rosa, Girasol o Tulipán) para iniciar la clasificación.")
