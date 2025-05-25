import os
os.environ["STREAMLIT_SERVER_PORT"] = os.getenv("PORT", "8501")
os.environ["STREAMLIT_SERVER_ADDRESS"] = "0.0.0.0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx
add_script_run_ctx()

import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch

# Configuración inicial
st.set_page_config(
    page_title="Análisis de Opiniones",
    layout="wide",
    menu_items={
        'Get Help': 'https://github.com/tu-usuario/tu-repo',
        'About': "App de análisis de sentimientos optimizada para Render"
    }
)

# Descarga de recursos NLTK
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Modelos optimizados (cache limitado)
@st.cache_resource(max_entries=2, show_spinner=False)
def load_models():
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",  # Modelo ligero
            device_map="auto",
            torch_dtype=torch.float16
        )
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        return pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )
    except Exception as e:
        st.error(f"Error cargando modelo: {str(e)}")
        return None

# Interfaz de usuario optimizada
def main():
    st.title("📊 Análisis de Opiniones Optimizado")
    
    analyzer = load_models()
    
    if analyzer is None:
        st.warning("El servicio está iniciando. Por favor espera 1 minuto y recarga.")
        return

    uploaded_file = st.file_uploader("Sube tu CSV con opiniones", type="csv")
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file).head(20)  # Limitar a 20 registros
        
        with st.spinner("Analizando (esto tomará 30 segundos)..."):
            # Procesamiento ligero
            st.subheader("Resultados Básicos")
            col1, col2 = st.columns(2)
            
            with col1:
                tokens = word_tokenize(' '.join(df.iloc[:, 0].astype(str)).lower())
                filtered = [w for w in tokens if w.isalpha() and w not in stopwords.words('spanish')]
                wordcloud = WordCloud(width=400, height=300).generate(' '.join(filtered))
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud)
                plt.axis('off')
                st.pyplot(plt, clear_figure=True)
            
            with col2:
                sample_text = df.iloc[0, 0][:200]  # Solo analizar muestra
                result = analyzer(sample_text)[0]
                st.metric("Sentimiento de muestra", 
                         f"{result['label']} ({result['score']:.0%})")

if __name__ == "__main__":
    main()