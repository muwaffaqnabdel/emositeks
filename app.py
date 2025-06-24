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
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = stemmer.stem(text)
    return text.strip()

# Fungsi validasi teks
def validate_text(text):
    if not text.strip():
        return False, "⚠️ Teks tidak boleh kosong."
    if len(text.strip()) < 3:
        return False, "⚠️ Teks terlalu pendek. Minimal 3 karakter."
    if text.strip().isdigit():
        return False, "⚠️ Teks tidak boleh hanya berisi angka."
    if not re.search(r'[a-zA-Z]', text):
        return False, "⚠️ Teks harus mengandung huruf untuk dapat dianalisis emosinya."
    if len(text) > 1000:
        return False, "⚠️ Teks terlalu panjang. Maksimal 1000 karakter."
    processed = preprocess(text)
    if not processed or len(processed.strip()) < 2:
        return False, "⚠️ Teks tidak mengandung kata yang dapat dianalisis."
    return True, None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']

    if model is None:
        return render_template('index.html', text=text, prediction=None,
                               alert="❌ Model tidak dapat dimuat. Periksa file model.")

    is_valid, error_message = validate_text(text)
    if not is_valid:
        return render_template('index.html', text=text, prediction=None,
                               alert=error_message)

    try:
        processed_text = preprocess(text)
        prediction = model.predict([processed_text])[0]

        if prediction is None or prediction == '':
            return render_template('index.html', text=text, prediction=None,
                                   alert="⚠️ Emosi tidak dapat dideteksi dari teks ini.")

        confidence = None
        try:
            proba = model.predict_proba([processed_text])[0]
            confidence = round(max(proba) * 100, 2)
        except:
            confidence = None

        if confidence is not None:
            if confidence >= 70:
                confidence_level = "high"
            elif confidence >= 50:
                confidence_level = "medium"
            else:
                confidence_level = "low"
            confidence_level_class = confidence_level.replace(" ", "-")
        else:
            confidence_level = None
            confidence_level_class = None

        return render_template('index.html', text=text, prediction=prediction,
                               confidence=confidence,
                               confidence_level=confidence_level,
                               confidence_level_class=confidence_level_class)

    except Exception as e:
        print(f"Error saat prediksi: {str(e)}")
        return render_template('index.html', text=text, prediction=None,
                               alert="❌ Terjadi kesalahan saat menganalisis emosi. Silakan coba lagi.")

if __name__ == '__main__':
    app.run(debug=True)
