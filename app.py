import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import zipfile

# Configuración de la página
st.set_page_config(page_title="Proyecto IA - Leandro Cortes", layout="wide")

# Cargar el modelo entrenado y las clases
model = tf.keras.models.load_model('best_model.keras')
class_names = np.loadtxt('clases.txt', dtype=str).tolist()

# Parámetros de la imagen
img_height = 180
img_width = 180

# Función para cargar y preprocesar una imagen
def load_and_preprocess_image(image_path, img_height, img_width):
    img = Image.open(image_path).resize((img_height, img_width))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, 0)  # Crear un batch
    return img_array

def main():
    # Título y subtítulo
    st.title("Proyecto IA - Leandro Cortes")
    st.subheader("Reconocimiento de Productos de Supermercado")

    # Información en la barra lateral izquierda
    st.sidebar.title("Información del Proyecto")
    st.sidebar.info("""
        **Nombre:** Leandro Cortes
        **Proyecto:** IA para Reconocimiento de Productos de Supermercado
        **Productos Reconocidos:**
        - Leche / MILK
        - Banana 
        - Avocado 
        - Manzana / Apple
        - Gaseosas / Sodas
        - Paquetes de papas / Chips
        - Galletas / Biscuits
    """)

    st.sidebar.title("¿Cómo funciona?")
    st.sidebar.info("""
        - Puedes subir una imagen desde tu dispositivo.
        - El modelo de IA analizará la imagen y predecirá a qué clase de producto pertenece.
    """)

    # Layout para incluir una barra lateral derecha
    col1, col2 = st.columns([3, 1])

    with col1:
        # Opción para subir una imagen
        uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            try:
                # Preprocesar la imagen
                img_array = load_and_preprocess_image(uploaded_file, img_height, img_width)

                # Realizar la predicción
                predictions = model.predict(img_array)
                score = tf.nn.softmax(predictions[0])

                # Mostrar la imagen y la predicción
                img = Image.open(uploaded_file)
                st.image(img, caption="Imagen subida", use_column_width=True)
                st.write(
                    "Esta imagen pertenece a la clase {} con una confianza de {:.2f} %."
                    .format(class_names[np.argmax(score)], 100 * np.max(score))
                )
            except Exception as e:
                st.error(f"Error al procesar la imagen: {e}")

    with col2:
        st.title("Información adicional")
        st.image("supermercado.jpg", caption="Supermercado", use_column_width=True)
        st.write("""
        **Fuente de las imágenes:**
        - Google Images
        - Kaggle Datasets
        - Roboflow
        - Imágenes propias
        """)

   if st.sidebar.button('Descargar imágenes de prueba'):
        zip_file = 'imagenes_prueba.zip'
        with zipfile.ZipFile(zip_file, 'w') as zf:
            for folder, subfolders, files in os.walk('path_to_your_images'):
                for file in files:
                    zf.write(os.path.join(folder, file), os.path.relpath(os.path.join(folder, file), 'path_to_your_images'))
        st.sidebar.success(f"Imágenes de prueba descargadas: {zip_file}")
        with open(zip_file, "rb") as f:
            st.sidebar.download_button('Descargar imágenes de prueba', f, file_name=zip_file)

if __name__ == "__main__":
    main()
