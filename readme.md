# 📺 YouTube Comment Sentiment Analysis with IndoBERT

Proyek ini adalah aplikasi berbasis web yang digunakan untuk menganalisis sentimen komentar pada video YouTube secara *real-time*. Aplikasi ini menggunakan model **IndoBERT** yang telah di-*fine-tune* untuk memahami nuansa bahasa Indonesia, termasuk penanganan emoji dan pembersihan teks otomatis.



## 🌟 Fitur Utama
- **Video Integration**: Menampilkan pemutar video YouTube langsung di dalam aplikasi.
- **Automated Scraping**: Mengambil komentar ulasan secara otomatis berdasarkan URL video.
- **Advanced Preprocessing**: 
    - Pembersihan URL, Mention, dan Hashtag.
    - Penanganan tanda baca berulang (intensitas emosi).
    - **Emoji Support**: Mengubah emoji menjadi deskripsi teks agar dapat dipahami model.
- **IndoBERT Analysis**: Mengklasifikasikan sentimen menjadi **Positif**, **Negatif**, atau **Netral**.
- **Data Visualization**: Menampilkan distribusi sentimen dalam bentuk tabel dan grafik batang interaktif.

## 🛠️ Arsitektur Teknologi
Aplikasi ini dibangun menggunakan *stack* teknologi Data Science modern:
- **Python 3.x**: Bahasa pemrograman utama.
- **Gradio**: Framework untuk antarmuka pengguna (UI) web.
- **Hugging Face Transformers**: Library untuk menjalankan model IndoBERT.
- **Pandas**: Manipulasi dan pengolahan data tabel.
- **Matplotlib**: Visualisasi data statistik.
- **Emoji Library**: Konversi emoticon menjadi teks deskripsi.

## 📋 Alur Kerja (Pipeline)
Proyek ini mengikuti standar siklus hidup data science:
1. **Load**: Mengambil data mentah (raw comments) via YouTube API.
2. **Preprocess**: Tahap *data cleaning* menggunakan Regex dan normalisasi emoji.
3. **Process**: Inferensi data menggunakan model Transformer (IndoBERT) untuk mendapatkan label sentimen dan skor kepercayaan (*confidence score*).
4. **Visualize**: Agregasi hasil dan pembuatan laporan visual.



## 🚀 Cara Instalasi

1. **Clone Repositori**
   ```bash
   git clone https://github.com/gungsatya/sentimen-analysis-youtube.git
   cd sentimen-analysis-youtube
   ```

2. **Buat Virtual Environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   .venv\Scripts\activate     # Windows
   ```

3. **Instal Dependensi**
   ```bash
   pip install -r requirements.txt
   ```
   *Catatan: Pastikan Anda sudah menginstal `torch`, `transformers`, `gradio`, `pandas`, `matplotlib`, dan `emoji`.*

4. **Konfigurasi API**
   Pastikan Anda memiliki akses internet untuk mengunduh model IndoBERT secara otomatis saat aplikasi dijalankan pertama kali.

## 💻 Cara Penggunaan
1. Jalankan aplikasi:
   ```bash
   python src/app.py
   ```
2. Buka browser dan akses alamat yang tertera (biasanya `http://127.0.0.1:7860`).
3. Masukkan **URL Video YouTube**.
4. Klik tombol **1. Load** untuk menarik komentar.
5. Klik tombol **2. Preprocess** untuk membersihkan teks.
6. Klik tombol **3. Process** untuk melakukan analisis sentimen.

## 📊 Detail Model
Aplikasi ini secara default menggunakan model:
`mdhugol/indonesia-bert-sentiment-classification`

Model ini dipilih karena akurasinya yang tinggi dalam menangani teks non-formal bahasa Indonesia dan kemampuannya membedakan tiga kelas sentimen (Positif, Netral, Negatif).

---
**Tugas Proyek - Foundation of Data Science**
*Program Magister Teknologi Informasi*