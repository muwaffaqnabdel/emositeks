from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
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
print("Cleaning data...")
df.dropna(inplace=True)

# Pastikan label tidak ada spasi ekstra
df['label'] = df['label'].str.strip()

# Preprocessing
print("Preprocessing text data...")
df['text'] = df['text'].apply(preprocess)

# Hapus teks kosong setelah preprocessing
df = df[df['text'].str.len() > 0]

# Tampilkan info dataset
print(f"Dataset shape: {df.shape}")
print(f"Label distribution:\n{df['label'].value_counts()}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

# Buat pipeline dengan parameter yang lebih baik
print("Training model...")
model = make_pipeline(
    TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),  # Unigram dan bigram
        min_df=2,           # Minimal muncul di 2 dokumen
        max_df=0.95         # Maksimal muncul di 95% dokumen
    ),
    LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight='balanced'  # Handle imbalanced data
    )
)

model.fit(X_train, y_train)

# Evaluasi model
print("Evaluating model...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Simpan model
print("Saving model...")
joblib.dump(model, 'emotion_model.pkl')
print("Model berhasil disimpan ke emotion_model.pkl")