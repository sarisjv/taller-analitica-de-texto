import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
import nltk
from nltk.corpus import stopwords
from transformers import pipeline
import time

# Configuración básica
st.set_page_config(
    page_title="Analizador Ligero",
    layout="wide",
    menu_items={
        'Get Help': 'https://github.com/tu-usuario/tu-repo',
        'About': "App optimizada para Render Free Tier"
    }
)

# Descarga mínima de NLTK
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Modelos optimizados con caché
@st.cache_resource(ttl=3600, show_spinner=False)
def load_light_model():
    return pipeline(
        "text-classification",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=-1  # Usar CPU
    )

# Datos de ejemplo
opiniones = [
    "Producto excelente, lo recomiendo totalmente.",
    "No cumple con lo prometido, muy decepcionante.",
    # ... (agrega tus 20 opiniones aquí)
]

# Funciones optimizadas
def quick_clean(text):
    return ' '.join([w.lower() for w in nltk.word_tokenize(text) 
                   if w.isalpha() and w not in stopwords.words('spanish')])

def main():
    st.title("Análisis de Opiniones Optimizado")
    
    # Pestañas para organización
    tab1, tab2 = st.tabs(["📊 Análisis Existente", "➕ Nuevo Análisis"])
    
    with tab1:
        st.header("Opiniones Almacenadas")
        
        # Wordcloud optimizado
        with st.spinner("Procesando opiniones..."):
            start_time = time.time()
            text = ' '.join(opiniones)
            tokens = quick_clean(text)[:5000]  # Limitar tamaño
            
            wc = WordCloud(width=600, height=300).generate(tokens)
            plt.figure(figsize=(10, 5))
            plt.imshow(wc)
            plt.axis('off')
            st.pyplot(plt, clear_figure=True)
            
            st.info(f"Procesado en {time.time()-start_time:.2f} segundos")
    
    with tab2:
        st.header("Analizar Nueva Opinión")
        user_input = st.text_area("Escribe tu comentario (máx. 200 caracteres):", max_chars=200)
        
        if user_input:
            model = load_light_model()
            result = model(user_input[:200])[0]  # Limitar entrada
            st.metric("Sentimiento", 
                     f"{result['label']} ({result['score']:.0%} confianza)")

if __name__ == "__main__":
    main()
            
