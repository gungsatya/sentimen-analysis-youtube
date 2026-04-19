import re
import os
import csv
import emoji
from nltk.tokenize import word_tokenize
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from transformers import pipeline

# Menggunakan model yang sudah di-fine-tune untuk sentimen Indonesia
# Gunakan model yang sudah terbukti bisa di-load
pretrained_model = "mdhugol/indonesia-bert-sentiment-classification"

try:
    # Inisialisasi pipeline classification
    sentiment_analysis = pipeline(
        "text-classification", 
        model=pretrained_model,
        tokenizer=pretrained_model
    )
except Exception as e:
    print(f"Error saat memuat model: {e}")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, 'data', 'slang_clean.csv')

stopword_factory = StopWordRemoverFactory()
stopwords_id = stopword_factory.get_stop_words()
stemmer_factory = StemmerFactory()
stemmer = stemmer_factory.create_stemmer()

slang_dict = {}

try:
    with open(CSV_PATH, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=';')
        next(reader, None)  # Melewati baris header
        for row in reader:
            if len(row) == 2:
                singkatan = row[0].strip().lower()
                original = row[1].strip().lower()
                slang_dict[singkatan] = original
except FileNotFoundError:
    print(f"Warning: File CSV kamus slang tidak ditemukan di path:\n{CSV_PATH}")

def get_safe_stopwords():
    # Daftar kata yang ingin dipertahankan (tidak dihapus sebagai stopword)
    safe_words = set([
        "tidak", "bukan", "belum", "jangan", "tanpa", "hanya", "saja", "cukup",
        "masih", "perlu", "harus", "akan", "mau", "ingin", "boleh", "dapat",
        "semua", "setiap", "beberapa", "banyak"
    ])
    
    # Filter stopwords untuk menghapus kata-kata yang ada di safe_words
    filtered_stopwords = [word for word in stopwords_id if word not in safe_words]
    
    return set(filtered_stopwords)

def normalize_slang(text):
    words = text.split()
    normalized_words = [slang_dict.get(word, word) for word in words]
    return ' '.join(normalized_words)

def clean_text(text):
    # 1. Ubah ke lowercase
    text = text.lower()
    
    # 2. Tangani Emoji: Ubah emoji menjadi teks deskripsi dalam bahasa Indonesia
    # Contoh: 😍 menjadi :wajah_tersenyum_dengan_mata_berbentuk_hati:
    text = emoji.demojize(text, language='id')
    
    # 3. Hapus URL, Mention, dan Hashtag
    text = re.sub(r'https?://\S+|www\.\S+', '', text) 
    text = re.sub(r'@\w+', '', text)                  
    text = re.sub(r'#\w+', '', text)                  
    
    # 4. Hapus karakter yang BUKAN alfanumerik, tanda baca standar, atau tanda titik dua (:)
    # Kita tambahkan ':' di regex agar label emoji dari demojize tidak terhapus
    text = re.sub(r'[^a-z0-9.,!?():_\s]', ' ', text)
    
    # 5. Hapus tanda baca yang diulang-ulang (misal: "!!!!!" -> "!")
    text = re.sub(r'([.,!?()])\1+', r'\1', text)
    
    # 6. Berikan spasi antara tanda baca agar tokenisasi BERT lebih bersih
    text = re.sub(r'([.,!?()])', r' \1 ', text)
    
    # 7. Hapus spasi berlebih dan trim
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def preprocess_text(text):

    # 1. Bersihkan teks (lowercase, hapus URL/mention/hashtag, karakter aneh, dll)
    text = clean_text(text)
    
    # 2. Unslang
    text = normalize_slang(text)
    
    # # 3. Tokenisasi
    # tokens = word_tokenize(text)
    
    # # 4. Stopword Removal
    # safe_stopwords = get_safe_stopwords()
    # tokens_no_stop = [word for word in tokens if word not in safe_stopwords]
    
    # # 5. Stemming (Mengembalikan kata dasar)
    # text_to_stem = ' '.join(tokens_no_stop)
    # processed_text = stemmer.stem(text_to_stem)
    
    # return processed_text

    return text


def analyze_sentiment(text):
    if not text.strip():
        return "Netral", 0.0
    
    try:
        result = sentiment_analysis(text[:1500])[0]  # Batasi panjang teks untuk menghindari error max_token BERT (512 tokens)
        label_raw = result['label']
        confidence = result['score']
        
        if label_raw == "LABEL_0":
            label_str = "Positif"
        elif label_raw == "LABEL_1":
            label_str = "Netral"
        elif label_raw == "LABEL_2":
            label_str = "Negatif"
        else:
            label_str = label_raw
            
        return label_str, confidence
    
    except Exception as e:
        print(f"Error during sentiment analysis: {e}")
        return "Netral", 0.0