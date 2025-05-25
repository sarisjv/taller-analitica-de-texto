import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import pandas as pd

# Configuración inicial
st.set_page_config(page_title="Análisis de Opiniones", layout="wide")
st.title("Análisis de Opiniones sobre Bases de Maquillaje")

# Descargar recursos de NLTK
nltk.download('punkt')
nltk.download('stopwords')

# Opiniones integradas en el código
opiniones = [
    "Un sérum magnífico, deja la piel espectacular con un acabado natural, el tono está muy bien. Si quieres una opción natural de maquillaje esta es la mejor.",
    "Este producto es maravilloso, minimiza imperfecciones con una sola aplicación al día. 10/10.",
    "Es la mejor base si buscas una cobertura muy natural. No se nota que traes algo puesto, pero empareja el tono y deja la piel luciendo muy sana y bonita.",
    "Excelente base buen cubrimiento.",
    "Mi piel es sensible y este producto es el mejor aliado del día a día, excelente cubrimiento, rendimiento porque con poco tienes sobre el rostro y te ves tan natural.",
    "Excelente base buen cubrimiento.",
    "El empaque es terrible, no la volveré a comprar porque no sirve el envase, el producto no sale por el aplicador, es fatal.",
    "Sí se siente una piel diferente después de usar el producto.",
    "Me gusta mucho cómo deja mi piel, es buen producto aunque no me gusta su presentación.",
    "Me parece buena, pero pienso que huele mucho a alcohol, no sé si es normal.",
    "Creo que fue el color que no lo supe elegir, no está mal, pero me imaginaba algo más uff.",
    "La base de maquillaje ofrece un acabado mate y aterciopelado que deja la piel lisa y es fácil de aplicar. En general, es una base que destaca por su buen desempeño y calidad.",
    "La base de maquillaje ofrece un acabado muy lindo y natural.",
    "Muy buen producto, solo que dura poco tiempo, por ahí unas 5 horas, pero muy bueno.",
    "Excelente cobertura y precio.",
    "No es para nada grasosa.",
    "El producto es mucho más oscuro de lo que aparece en la referencia.",
    "Pensé me sentaría mejor el número 8, es muy buena pero noto que toca como poner dos veces para mejor cobertura pero ya queda la piel pasteluda.",
    "No me gustó su cobertura.",
    "La sensación en la piel no me gusta, me arde al aplicarla."
]

# Función para limpiar y tokenizar texto
def clean_and_tokenize(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Záéíóúñ\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('spanish'))
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    return tokens

# Análisis de sentimientos simple
def analyze_sentiment(text):
    positive_words = ['magnífico', 'espectacular', 'maravilloso', 'excelente', 'buen', 'mejor', 'bonita', 'lindo', 'natural']
    negative_words = ['terrible', 'fatal', 'arde', 'pasteluda', 'oscuro', 'horas', 'alcohol']
    
    text = text.lower()
    pos = sum(1 for word in positive_words if word in text)
    neg = sum(1 for word in negative_words if word in text)
    
    if pos > neg:
        return "Positivo", pos/(pos+neg+1)
    elif neg > pos:
        return "Negativo", neg/(pos+neg+1)
    else:
        return "Neutral", 0.5

# Interfaz de usuario
def main():
    st.header("Opiniones Analizadas")
    
    # Convertir a DataFrame
    df = pd.DataFrame({'Opinión': opiniones})
    
    # Mostrar todas las opiniones
    if st.checkbox("Mostrar todas las opiniones"):
        st.dataframe(df)
    
    # Análisis de texto
    st.subheader("Análisis de Texto")
    col1, col2 = st.columns(2)
    
    with col1:
        # Nube de palabras
        st.write("**Nube de palabras**")
        all_text = ' '.join(opiniones)
        tokens = clean_and_tokenize(all_text)
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(tokens))
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)
    
    with col2:
        # Palabras más frecuentes
        st.write("**Palabras más frecuentes**")
        word_counts = Counter(tokens)
        top_words = word_counts.most_common(10)
        top_words_df = pd.DataFrame(top_words, columns=['Palabra', 'Frecuencia'])
        st.bar_chart(top_words_df.set_index('Palabra'))
    
    # Análisis de sentimientos
    st.subheader("Análisis de Sentimientos")
    df['Sentimiento'] = df['Opción'].apply(lambda x: analyze_sentiment(x)[0])
    df['Puntaje'] = df['Opción'].apply(lambda x: analyze_sentiment(x)[1])
    
    # Mostrar resultados
    st.dataframe(df[['Opción', 'Sentimiento', 'Puntaje']].sort_values('Puntaje', ascending=False))
    
    # Distribución de sentimientos
    st.write("**Distribución de sentimientos**")
    sentiment_counts = df['Sentimiento'].value_counts()
    st.bar_chart(sentiment_counts)

if __name__ == "__main__":
    main()
