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

# Fungsi validasi teks
def validate_text(text):
    """
    Validasi teks input untuk memastikan cocok untuk prediksi emosi
    Returns: (is_valid, error_message)
    """
    
    # Cek apakah teks kosong
    if not text.strip():
        return False, "⚠️ Teks tidak boleh kosong."
    
    # Cek panjang teks minimal
    if len(text.strip()) < 3:
        return False, "⚠️ Teks terlalu pendek. Minimal 3 karakter."
    
    # Cek apakah teks hanya berisi angka
    if text.strip().isdigit():
        return False, "⚠️ Teks tidak boleh hanya berisi angka."
    
    # Cek apakah teks mengandung huruf (bukan hanya simbol/angka)
    if not re.search(r'[a-zA-Z]', text):
        return False, "⚠️ Teks harus mengandung huruf untuk dapat dianalisis emosinya."
    
    # Cek apakah teks terlalu panjang (opsional)
    if len(text) > 1000:
        return False, "⚠️ Teks terlalu panjang. Maksimal 1000 karakter."
    
    # Cek apakah teks mengandung kata yang bermakna setelah preprocessing
    processed = preprocess(text)
    if not processed or len(processed.strip()) < 2:
        return False, "⚠️ Teks tidak mengandung kata yang dapat dianalisis."
    
    return True, None

# Route halaman utama
@app.route('/')
def home():
    return render_template('index.html')

# Route prediksi
@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    
    # Cek apakah model berhasil dimuat
    if model is None:
        return render_template('index.html', text=text, prediction=None,
                            alert="❌ Model tidak dapat dimuat. Periksa file model.")
    
    # Validasi teks input
    is_valid, error_message = validate_text(text)
    if not is_valid:
        return render_template('index.html', text=text, prediction=None,
                            alert=error_message)
    
    try:
        # Preprocessing teks
        processed_text = preprocess(text)
        
        # Prediksi emosi
        prediction = model.predict([processed_text])[0]
        
        # Cek apakah prediksi berhasil
        if prediction is None or prediction == '':
            return render_template('index.html', text=text, prediction=None,
                                alert="⚠️ Emosi tidak dapat dideteksi dari teks ini.")
        
        # Jika model support predict_proba, tampilkan confidence
        confidence = None
        confidence_level = "tinggi"
        try:
            proba = model.predict_proba([processed_text])[0]
            confidence = round(max(proba) * 100, 2)
            
            # Tentukan level confidence
            if confidence >= 70:
                confidence_level = "tinggi"
            elif confidence >= 50:
                confidence_level = "sedang"
            elif confidence >= 30:
                confidence_level = "rendah"
            else:
                confidence_level = "sangat rendah"
            
        except:
            # Model tidak support predict_proba
            confidence_level = "tidak diketahui"
        
        return render_template('index.html', text=text, prediction=prediction, 
                            confidence=confidence, confidence_level=confidence_level)
    
    except Exception as e:
        # Handle error saat prediksi
        print(f"Error saat prediksi: {str(e)}")
        return render_template('index.html', text=text, prediction=None,
                            alert="❌ Terjadi kesalahan saat menganalisis emosi. Silakan coba lagi.")

if __name__ == '__main__':
    app.run(debug=True)
