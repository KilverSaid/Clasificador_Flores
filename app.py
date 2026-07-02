import json
from pathlib import Path
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# Configuración inicial de la interfaz a pantalla ancha
st.set_page_config(
    page_title="Reconocimiento de Flores - IA", 
    layout="wide"
)

# Estilos visuales en la barra lateral
with st.sidebar:
    st.header("Panel de Control")
    st.markdown("---")
    st.info("Nota de entrega: Asegúrate de que el archivo 'flower_model.h5' esté ubicado en la raíz de este directorio.")
    
    # Sección de depuración colapsable
    with st.expander("Modo Depuración (Debug)", expanded=False):
        st.caption("Verifica que las clases coincidan exactamente con el orden alfabético del ImageDataGenerator de tu entrenamiento.")

# Encabezado principal
st.title("Clasificación Automatizada de Especies Botánicas")
st.markdown("### Clase: Inteligencia Artificial")
st.caption("Desarrollado por: **Kilver Said Nolasco Parada** | **Cuenta:** 20221930129")
st.markdown("---")

# Dimensiones estándar requeridas por MobileNetV2 y ruta del modelo
IMG_SIZE = (224, 224)
MODEL_PATH = Path("flower_model.h5")

# Mapeo y orden estricto de clases (sin emojis en las etiquetas)
CLASES_ENTRENAMIENTO = ["daisy", "dandelion", "rose", "sunflower", "tulip"]
LABELS_ES = {
    "daisy": "Margarita",
    "dandelion": "Diente de León",
    "rose": "Rosa",
    "sunflower": "Girasol",
    "tulip": "Tulipán"
}

@st.cache_resource
def cargar_modelo():
    if MODEL_PATH.exists():
        return tf.keras.models.load_model(MODEL_PATH, compile=False)
    st.error("Error crítico: No se detectó 'flower_model.h5'. Revisa tu repositorio.")
    st.stop()

def preparar_imagen(img):
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)

def predecir(img):
    probabilidades = modelo.predict(preparar_imagen(img), verbose=0)[0]
    top_indices = np.argsort(probabilidades)[::-1]
    return [
        (LABELS_ES[CLASES_ENTRENAMIENTO[i]], float(probabilidades[i]) * 100)
        for i in top_indices
    ]

# Carga del modelo entrenado
modelo = cargar_modelo()

# Diseño en dos columnas para el despliegue
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("Entrada de Datos")
    archivo = st.file_uploader(
        "Arrastra o selecciona la imagen de la flor a evaluar:", 
        type=["jpg", "jpeg", "png"]
    )
    
    if archivo:
        imagen = Image.open(archivo)
        st.image(imagen, caption="Vista previa del espécimen", use_container_width=True)
    else:
        st.warning("Esperando archivo... Por favor sube una imagen para activar la red neuronal.")

with col2:
    st.subheader("Diagnóstico del Modelo")
    
    if archivo:
        with st.spinner("Analizando patrones con MobileNetV2..."):
            resultados = predecir(imagen)
        
        # Cuadro de resultado destacado
        st.metric(
            label="Predicción Óptima Detectada", 
            value=resultados[0][0], 
            delta=f"{resultados[0][1]:.2f}% de Confianza"
        )
        
        st.markdown("#### Distribución de Probabilidades:")
        for clase, prob in resultados:
            st.write(f"**{clase}**")
            st.progress(prob / 100.0)
    else:
        st.info("Tip: El modelo reconoce con precisión Margaritas, Dientes de León, Rosas, Girasoles y Tulipanes.")
