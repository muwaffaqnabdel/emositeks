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
    # Normalisasi kata slang
    slang_dict = {
        'gajian': 'mendapat gaji senang',
        'belanja': 'membeli senang',
        'membelikan': 'membeli senang'
    }
    
    for slang, formal in slang_dict.items():
        text = text.replace(slang, formal)
    
    text = re.sub(r'[^\w\s]', '', text)  # Hilangkan tanda baca
    text = re.sub(r'\s+', ' ', text).strip()  # Hilangkan spasi berlebih
    return stemmer.stem(text.lower())  # Lowercase + stemming

# Daftar kata kuat yang sangat mewakili emosi
strong_keywords = {
    'marah': ['marah', 'kesal', 'emosi', 'murka', 'geram', 'jengkel', 'dongkol', 'sewot', 'naik pitam', 'memukul', 'memaki', 'mencaci', 'berteriak', 'amarah', 'mengumpat', 'melawan', 'protes', 'marah-marah', 'berantem', 'berkelahi'],
    'sedih': ['sedih', 'menangis', 'kecewa', 'terpuruk', 'duka', 'galau', 'hancur', 'patah hati', 'larut', 'mellow', 'putus asa', 'depresi', 'murung', 'lesu', 'down', 'drop', 'terpukul', 'nestapa', 'sendu', 'melankolis'],
    'takut': ['takut', 'cemas', 'khawatir', 'panik', 'ngeri', 'was-was', 'deg-degan', 'gemetar', 'tegang', 'gugup', 'nervous', 'fobia', 'paranoid', 'resah', 'gelisah', 'stress', 'overthinking', 'waswas', 'horror', 'mengerikan'],
    'senang': ['senang', 'bahagia', 'gembira', 'ceria', 'suka', 'gajian', 'gaji', 'membeli', 'belanja', 'dapat', 'antusias', 'excited', 'girang', 'riang', 'sukacita', 'bergembira', 'tertawa', 'haha', 'hehe', 'yeay', 'mantap', 'keren', 'asik', 'wow', 'amazing', 'luar biasa', 'fantastis', 'hebat', 'bagus', 'menyenangkan', 'menggembirakan'],
    'cinta': ['cinta', 'sayang', 'kasih', 'rindu', 'jatuh hati', 'naksir', 'crush', 'gebetan', 'pacar', 'kekasih', 'beloved', 'honey', 'dear', 'romantis', 'mesra', 'intim', 'passionate', 'tergila-gila', 'terpesona', 'kasmaran'],
    'benci': ['benci', 'muak', 'jijik', 'tidak suka', 'anti', 'sebel', 'gondok', 'ogah', 'males', 'enggan', 'menolak', 'menyesal', 'kapok', 'trauma', 'antipati', 'alergi', 'muak banget', 'gak tahan', 'pengen muntah'],
    'terkejut': ['kaget', 'terkejut', 'syok', 'terkaget', 'surprise', 'wow', 'wah', 'astaga', 'ya ampun', 'oh my god', 'tidak percaya', 'mengejutkan', 'tak terduga', 'mendadak', 'tiba-tiba', 'unexpected', 'luar biasa', 'fantastis', 'incredible', 'unbelievable']
}

def get_manual_confidence(text):
    text = text.lower()

    # Kata netral/umum yang tidak langsung menunjukkan emosi
    neutral_keywords = ['makan', 'pergi', 'tugas', 'kerja', 'kuliah']

    # Cek apakah teks mengandung kata kuat
    for emotion, keywords in strong_keywords.items():
        for word in keywords:
            if word in text:
                return 85  # Confidence tinggi untuk kata kuat

    # Cek apakah teks netral
    for word in neutral_keywords:
        if word in text:
            return 50  # Netral, confidence = 50%

    return 0  # Tidak mengandung informasi emosi yang jelas

def rule_based_emotion(text):
    """Rule-based emotion detection sebagai fallback"""
    text_lower = text.lower()
    
    # Loop melalui semua emosi dan kata kuncinya
    for emotion, keywords in strong_keywords.items():
        if any(word in text_lower for word in keywords):
            return emotion, 80
    
    return '(kata tidak dapat di prediksi)', 0

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
        try:
            proba = model.predict_proba([processed_text])[0]
            confidence = round(max(proba) * 100, 2)
            
            # Jika confidence ML rendah, gunakan rule-based
            if confidence < 50:
                rule_emotion, rule_conf = rule_based_emotion(text)
                prediction = rule_emotion
                confidence = rule_conf
            
        except:
            # Model tidak support predict_proba, gunakan manual confidence
            confidence = get_manual_confidence(text)
            if confidence < 50:
                rule_emotion, rule_conf = rule_based_emotion(text)
                prediction = rule_emotion
                confidence = rule_conf
            
        # Tentukan level confidence
        if confidence >= 70:
            confidence_level = "high"
        elif confidence >= 50:
            confidence_level = "medium"
        else:
            confidence_level = "low"

        confidence_level_class = confidence_level.replace(" ", "-")

        return render_template('index.html', text=text, prediction=prediction,
                                confidence=confidence,
                                confidence_level=confidence_level,
                                confidence_level_class=confidence_level_class)
    
    except Exception as e:
        # Handle error saat prediksi
        print(f"Error saat prediksi: {str(e)}")
        return render_template('index.html', text=text, prediction=None,
                            alert="❌ Terjadi kesalahan saat menganalisis emosi. Silakan coba lagi.")

if __name__ == '__main__':
    app.run(debug=True)
