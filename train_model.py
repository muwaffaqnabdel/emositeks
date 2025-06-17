from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import pandas as pd
import joblib
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Inisialisasi stemmer Bahasa Indonesia
stemmer = StemmerFactory().create_stemmer()

# Fungsi preprocessing
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return stemmer.stem(text)

# Load dataset
df = pd.read_csv('data/Emotion_classify_Data.csv')
df['text'] = df['text'].apply(preprocess)

# Buat pipeline TF-IDF + Naive Bayes
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(df['text'], df['label'])

# Simpan model ke file
joblib.dump(model, 'emotion_model.pkl')
print("Model berhasil disimpan ke emotion_model.pkl")
