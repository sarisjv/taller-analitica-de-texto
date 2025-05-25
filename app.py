import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["STREAMLIT_SERVER_PORT"] = os.getenv("PORT", "8501")
os.environ["STREAMLIT_SERVER_ADDRESS"] = "0.0.0.0"

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import gc  # Garbage collector

# Configuración mínima
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Análisis de sentimientos sin transformers (alternativa liviana)
def simple_sentiment_analysis(text):
    positive_words = ['bueno', 'excelente', 'maravilloso', 'recomiendo', 'perfecto']
    negative_words = ['malo', 'terrible', 'horrible', 'pésimo', 'decepcionante']
    
    text = text.lower()
    pos = sum(1 for word in positive_words if word in text)
    neg = sum(1 for word in negative_words if word in text)
    
    if pos > neg:
        return "Positivo", pos/(pos+neg+1)
    elif neg > pos:
        return "Negativo", neg/(pos+neg+1)
    else:
        return "Neutral", 0.5

def main():
    st.title("📊 Análisis Ligero de Opiniones")
    
    uploaded_file = st.file_uploader("Sube tu archivo CSV", type="csv")
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file).head(15)  # Limitar a 15 registros
            
            # Análisis básico sin carga pesada
            st.subheader("Resultados")
            
            # Wordcloud simplificado
            text = ' '.join(df.iloc[:, 0].astype(str))
            tokens = [word.lower() for word in word_tokenize(text) 
                     if word.isalpha() and word not in stopwords.words('spanish')]
            
            if tokens:
                plt.figure(figsize=(10, 5))
                wordcloud = WordCloud(width=800, height=400).generate(' '.join(tokens))
                plt.imshow(wordcloud)
                plt.axis('off')
                st.pyplot(plt)
                plt.close()  # Liberar memoria
                
                # Análisis de sentimiento muestra
                sample = df.iloc[0, 0][:200]  # Solo primera opinión
                sentiment, score = simple_sentiment_analysis(sample)
                st.metric("Sentimiento muestra", f"{sentiment} ({score:.0%})")
            
            gc.collect()  # Forzar liberación de memoria
            
        except Exception as e:
            st.error(f"Error procesando archivo: {str(e)}")

if __name__ == "__main__":
    main()
