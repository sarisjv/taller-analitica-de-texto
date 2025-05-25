import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

# Configuración inicial
st.set_page_config(page_title="Análisis Avanzado de Opiniones", layout="wide", page_icon="📊")
st.title("📊 Analizador de Opiniones con IA")

# Descargar recursos de NLTK
nltk.download('punkt')
nltk.download('stopwords')

# Cargar modelos de lenguaje (con caché para optimización)
@st.cache_resource
def load_models():
    try:
        # Modelo para análisis de sentimientos
        sentiment_model = pipeline(
            "sentiment-analysis",
            model="finiteautomata/bertweet-base-sentiment-analysis",
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Modelo para resumen
        summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=0 if torch.cuda.is_available() else -1
        )
        
        return sentiment_model, summarizer
    except Exception as e:
        st.error(f"Error cargando modelos: {str(e)}")
        return None, None

sentiment_model, summarizer = load_models()

# Opiniones de ejemplo integradas
opiniones = [
    "Un sérum magnífico, deja la piel espectacular con un acabado natural, el tono está muy bien.",
    "Este producto es maravilloso, minimiza imperfecciones con una sola aplicación al día.",
    "Es la mejor base si buscas una cobertura muy natural. No se nota que traes algo puesto.",
    "Excelente base buen cubrimiento.",
    "Mi piel es sensible y este producto es el mejor aliado del día a día.",
    "Excelente base buen cubrimiento.",
    "El empaque es terrible, no la volveré a comprar porque no sirve el envase.",
    "Sí se siente una piel diferente después de usar el producto.",
    "Me gusta mucho cómo deja mi piel, es buen producto aunque no me gusta su presentación.",
    "Me parece buena, pero pienso que huele mucho a alcohol, no sé si es normal.",
    "Creo que fue el color que no lo supe elegir, no está mal, pero me imaginaba algo más.",
    "La base ofrece un acabado mate y aterciopelado que deja la piel lisa.",
    "La base de maquillaje ofrece un acabado muy lindo y natural.",
    "Muy buen producto, solo que dura poco tiempo, por ahí unas 5 horas.",
    "Excelente cobertura y precio.",
    "No es para nada grasosa.",
    "El producto es mucho más oscuro de lo que aparece en la referencia.",
    "Pensé me sentaría mejor el número 8, es muy buena pero noto que toca como poner dos veces.",
    "No me gustó su cobertura.",
    "La sensación en la piel no me gusta, me arde al aplicarla."
]

# Funciones de análisis
def clean_text(text):
    text = re.sub(r'[^a-zA-ZáéíóúñÁÉÍÓÚÑ\s]', '', text.lower())
    return ' '.join([word for word in word_tokenize(text) 
                    if word not in stopwords.words('spanish') and len(word) > 2])

def analyze_sentiment(text):
    if sentiment_model is None:
        return "Modelo no disponible", 0.0
    try:
        result = sentiment_model(text[:512])[0]
        return {"POS": "Positivo", "NEG": "Negativo", "NEU": "Neutral"}.get(result['label'], "Neutral"), result['score']
    except:
        return "Error", 0.0

def generate_summary(text):
    if summarizer is None:
        return "Modelo de resumen no disponible"
    try:
        return summarizer(text[:1024], max_length=130, min_length=30, do_sample=False)[0]['summary_text']
    except:
        return "Error generando resumen"

# Interfaz principal
def main():
    tab1, tab2 = st.tabs(["📝 Analizar Nuevos Comentarios", "📊 Explorar Opiniones Existentes"])
    
    with tab1:
        st.header("Analizar Nuevo Comentario")
        user_input = st.text_area("Escribe tu comentario sobre el producto:", height=150)
        
        if user_input:
            col1, col2 = st.columns(2)
            with col1:
                with st.spinner("Analizando sentimiento..."):
                    sentiment, score = analyze_sentiment(user_input)
                    st.metric("Sentimiento", f"{sentiment} ({score:.0%} confianza)")
            
            with col2:
                with st.spinner("Generando resumen..."):
                    summary = generate_summary(user_input)
                    st.text_area("Resumen:", value=summary, height=100)
    
    with tab2:
        st.header("Explorar 20 Opiniones de Ejemplo")
        df = pd.DataFrame({'Opinión': opiniones})
        
        # Análisis colectivo
        st.subheader("Análisis Colectivo")
        option = st.radio("Selecciona análisis:", 
                         ["🔍 Temas principales", "📌 Resumen general", "📈 Distribución de sentimientos"])
        
        if option == "🔍 Temas principales":
            all_text = ' '.join(opiniones)
            cleaned = clean_text(all_text)
            wordcloud = WordCloud(width=800, height=400).generate(cleaned)
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud)
            plt.axis('off')
            st.pyplot(plt)
            
            # Palabras más frecuentes
            words = Counter(cleaned.split()).most_common(10)
            st.write("**Palabras clave más frecuentes:**")
            for word, count in words:
                st.write(f"- {word} ({count} veces)")
        
        elif option == "📌 Resumen general":
            with st.spinner("Generando resumen de todas las opiniones..."):
                combined = " ".join([o[:200] for o in opiniones])  # Limitar longitud
                summary = generate_summary(combined)
                st.write(summary)
        
        elif option == "📈 Distribución de sentimientos":
            with st.spinner("Analizando sentimientos..."):
                df['Sentimiento'] = df['Opinión'].apply(lambda x: analyze_sentiment(x)[0])
                st.bar_chart(df['Sentimiento'].value_counts())
                
                # Mostrar ejemplos
                st.write("**Ejemplos por categoría:**")
                for sentiment in ["Positivo", "Neutral", "Negativo"]:
                    examples = df[df['Sentimiento'] == sentiment]['Opinión'].head(2)
                    if not examples.empty:
                        st.write(f"**{sentiment}:**")
                        for example in examples:
                            st.write(f"- {example[:100]}...")

if __name__ == "__main__":
    main()
