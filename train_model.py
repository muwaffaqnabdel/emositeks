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

# Bersihkan data kosong
df.dropna(inplace=True)

# Pastikan label tidak ada spasi ekstra
df['label'] = df['label'].str.strip()

# Preprocessing
df['text'] = df['text'].apply(preprocess)

# Buat pipeline TF-IDF + Naive Bayes
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(df['text'], df['label'])

# Simpan model
joblib.dump(model, 'emotion_model.pkl')
# Simpan vectorizer-nya juga jika diperlukan
joblib.dump(model.named_steps['tfidfvectorizer'], 'vectorizer.pkl')

print("Model berhasil disimpan ke emotion_model.pkl")
