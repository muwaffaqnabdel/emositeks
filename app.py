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
    text = re.sub(r'[^\w\s]', '', text)  # Hilangkan tanda baca
    text = re.sub(r'\s+', ' ', text).strip()  # Hilangkan spasi berlebih
    return stemmer.stem(text.lower())  # Lowercase + stemming

# Route halaman utama
@app.route('/')
def home():
    return render_template('index.html')

# Route prediksi
@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    

    if not text.strip():
        return render_template('index.html', text=text, prediction=None,
                               alert="⚠️ Teks tidak boleh kosong.")

    processed_text = preprocess(text)
    prediction = model.predict([processed_text])[0]

    return render_template('index.html', text=text, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
