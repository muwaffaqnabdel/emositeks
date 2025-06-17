from flask import Flask, request, render_template
import joblib
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

app = Flask(__name__)

# Load model
model = joblib.load('emotion_model.pkl')

# Inisialisasi stemmer
stemmer = StemmerFactory().create_stemmer()

# Fungsi preprocessing
def preprocess(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return stemmer.stem(text.lower())

# Route halaman utama
@app.route('/')
def home():
    return render_template('index.html')

# Route prediksi
@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    processed_text = preprocess(text)

    # Cek apakah input ada di dataset
    if processed_text not in dataset_texts:
        return render_template('index.html', text=text, prediction=None,
                               alert="⚠️ Teks tidak dikenali oleh model! Silakan masukkan teks lain.")

    prediction = model.predict([processed_text])[0]
    return render_template('index.html', text=text, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
